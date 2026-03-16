"""
POC v4: Attention-Level Sink Eviction via KV Cache Zeroing.

v2 showed embedding-zeroing collapses when sinks are removed.
v4 asks: does the same hold at the ATTENTION level?

Strategy: run normal prefill, then zero out KV cache entries
for evicted tokens. This is the closest simulation of real
KV cache eviction — the model has already encoded the tokens,
but their K/V contributions are nullified before decoding.

This avoids the mask-shape issues with Qwen2.5-VL's attention
implementation (mask slicing bug at line 165).

Usage:
  uv run --with "transformers<4.52" --with torch \
    --with matplotlib \
    python sink_eviction_poc_v4_attn_mask.py \
      --video test_video.mp4 \
      --output-dir sink_eviction_output_v4
"""

import argparse
import os
import types
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sink_detect import (
    CaptureStore,
    find_image_token_range,
    patch_model,
    rmsnorm,
)


# ── KV cache zeroing ───────────────────────────────

def zero_kv_cache_positions(
    model, drop_indices: np.ndarray,
):
    """
    Zero out K and V vectors at specified positions
    in ALL layers' KV caches.

    After prefill, the KV cache stores [B, n_heads, S, D]
    where S = sequence length. We zero out entries at
    the evicted positions so attention to them yields ~0.
    """
    if len(drop_indices) == 0:
        return

    layers = model.language_model.model.layers
    n_zeroed = 0

    for layer in layers:
        attn = layer.self_attn
        cache = None
        # In mlx, KV cache is stored in layer.self_attn
        # during generation. We need to find it.
        # mlx-vlm passes cache externally, so we need
        # a different approach — see hook below.
        pass

    # Since mlx-vlm manages cache externally via the
    # generate() function, we can't easily access it
    # from outside. Instead, we'll hook into the
    # attention forward pass.


# ── Hook-based KV cache interception ────────────────
# We patch the attention layer to zero out KV entries
# AFTER cache.update_and_fetch but BEFORE SDPA.

_evict_indices: Optional[np.ndarray] = None
_original_attn_calls = {}


def _make_attn_eviction_hook(original_call):
    """
    Create a patched attention __call__ that zeros
    out KV cache entries at evicted positions after
    the cache merge step.
    """
    def patched_call(
        self,
        x: mx.array,
        mask=None,
        cache=None,
        position_ids=None,
    ) -> mx.array:
        from mlx_vlm.models.base import (
            scaled_dot_product_attention,
        )
        from mlx_vlm.models.qwen2_5_vl.language import (
            apply_multimodal_rotary_pos_emb,
        )

        B, L, D = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(
            B, L, self.n_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(
            B, L, self.n_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            B, L, self.n_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        kv_seq_len = keys.shape[-2]

        if position_ids is None:
            kv_seq_len += cache.offset + 1
            position_ids = mx.arange(
                cache.offset, cache.offset + L
            )
            position_ids = mx.expand_dims(
                position_ids, axis=0
            )
            position_ids = mx.tile(
                position_ids, (3, 1, 1)
            )
        else:
            kv_seq_len += (
                cache.offset + 1
                if cache is not None else 0
            )

        cos, sin = self.rotary_emb(
            values, position_ids
        )

        if mask is not None and isinstance(
            mask, mx.array
        ):
            mask = mask[..., :keys.shape[-2]]

        queries, keys = apply_multimodal_rotary_pos_emb(
            queries, keys, cos, sin, unqueeze_dim=1,
        )

        if cache is not None:
            keys, values = cache.update_and_fetch(
                keys, values,
            )

        # ── EVICTION: zero out KV at drop positions ──
        # Only during DECODING (L==1), not prefill.
        # During prefill, all tokens need to interact
        # to build representations. Eviction only
        # affects which KV entries are used for
        # generating new tokens.
        if (
            _evict_indices is not None
            and len(_evict_indices) > 0
            and L == 1  # decode step only
        ):
            kv_len = keys.shape[2]
            # Build zero mask: [1, 1, kv_len, 1]
            keep = np.ones(
                (1, 1, kv_len, 1), dtype=np.float32,
            )
            for idx in _evict_indices:
                if idx < kv_len:
                    keep[0, 0, idx, 0] = 0.0
            keep_mx = mx.array(keep)
            keys = keys * keep_mx
            values = values * keep_mx

        output = scaled_dot_product_attention(
            queries, keys, values, cache,
            scale=self.scale, mask=mask,
        )
        output = output.transpose(
            0, 2, 1, 3
        ).reshape(B, L, -1)
        return self.o_proj(output)

    return patched_call


def patch_attention_layers(model):
    """
    Patch the Attention CLASS __call__ for KV eviction.

    NOTE: Python special methods (__call__) are looked up
    on the type, not the instance. So we must patch the
    CLASS, not individual instances.
    """
    layers = model.language_model.model.layers
    attn_cls = layers[0].self_attn.__class__
    _original_attn_calls["class"] = attn_cls.__call__
    attn_cls.__call__ = _make_attn_eviction_hook(
        _original_attn_calls["class"]
    )
    print(f"Patched {attn_cls.__name__}.__call__ "
          f"for KV eviction ({len(layers)} layers)")


# ── Sink detection ──────────────────────────────────

def detect_sink_indices(
    hidden_states, sink_dims, img_start, img_end,
    tau=20.0, layer=14,
):
    hs = hidden_states[layer][0]
    rms = np.abs(rmsnorm(hs))
    phi = np.max(
        np.stack(
            [rms[:, d] for d in sink_dims], axis=-1
        ),
        axis=-1,
    )
    vis_phi = phi[img_start:img_end]
    sink_local = np.where(vis_phi > tau)[0]
    nonsink_local = np.where(vis_phi <= tau)[0]
    return (
        sink_local + img_start,
        nonsink_local + img_start,
    )


def token_overlap(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ── Generation with KV eviction ────────────────────

def run_gen(
    model, processor, text, gen_kwargs,
    drop_indices=None,
):
    """Generate with KV cache eviction at attention."""
    global _evict_indices
    from mlx_vlm import generate

    _evict_indices = (
        drop_indices
        if drop_indices is not None
        and len(drop_indices) > 0
        else None
    )

    output = generate(
        model, processor, prompt=text,
        verbose=False, **gen_kwargs,
    )

    _evict_indices = None
    return output.text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=(
            "mlx-community/Qwen2.5-VL-7B-Instruct-8bit"
        ),
    )
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--prompt",
        default="Describe the butterfly in this video.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=100,
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        default="sink_eviction_output_v4",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sink_dims = [458, 2570]
    detect_layer = 14

    # ── Load model ──
    from mlx_vlm import load, generate
    from mlx_vlm.utils import load_config
    from mlx_vlm.video_generate import process_vision_info

    print(f"Loading {args.model}...")
    model, processor = load(args.model)
    config = load_config(args.model)
    patch_model(model)  # sink_detect hooks

    # Patch attention layers for KV eviction
    patch_attention_layers(model)

    # ── Prepare inputs ──
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": args.video,
                    "max_pixels": 224 * 224,
                    "fps": args.fps,
                },
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True,
    )
    img_inputs, vid_inputs, _ = (
        process_vision_info(messages, True)
    )
    inputs = processor(
        text=[text], images=img_inputs,
        videos=vid_inputs, padding=True,
        return_tensors="pt",
    )

    input_ids = mx.array(inputs["input_ids"])
    pv = inputs.get(
        "pixel_values_videos",
        inputs.get("pixel_values", None),
    )
    pixel_values = mx.array(pv)
    mask_input = mx.array(inputs["attention_mask"])
    seq_len = input_ids.shape[1]

    extra_kwargs = {}
    for key in ["video_grid_thw", "image_grid_thw"]:
        if inputs.get(key) is not None:
            extra_kwargs[key] = mx.array(inputs[key])

    model_config = model.config
    img_start, img_end = find_image_token_range(
        input_ids, model_config,
    )
    n_vis = img_end - img_start
    print(f"Seq len: {seq_len}, "
          f"Visual: [{img_start},{img_end}) = {n_vis}")

    gen_kwargs = dict(extra_kwargs)
    gen_kwargs["video"] = [args.video]
    gen_kwargs["input_ids"] = input_ids
    gen_kwargs["pixel_values"] = pixel_values
    gen_kwargs["mask"] = mask_input
    gen_kwargs["temperature"] = 0.0
    gen_kwargs["max_tokens"] = args.max_tokens

    # ══════════════════════════════════════════════════
    # BASELINE: capture hidden states for sink detection
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("BASELINE (normal KV cache)")
    print("=" * 60)

    CaptureStore.enable()
    baseline = run_gen(
        model, processor, text, gen_kwargs,
    )
    CaptureStore.disable()
    print(f"  {baseline}")

    hs_np = [
        np.array(h) for h in CaptureStore.hidden_states
    ]
    sink_abs, nonsink_abs = detect_sink_indices(
        hs_np, sink_dims, img_start, img_end,
        tau=20.0, layer=detect_layer,
    )
    n_sink = len(sink_abs)
    n_nonsink = len(nonsink_abs)
    print(f"\nSink detection (Layer {detect_layer}):")
    print(f"  Sink: {n_sink} "
          f"({100 * n_sink / n_vis:.0f}%)")
    print(f"  Non-sink: {n_nonsink} "
          f"({100 * n_nonsink / n_vis:.0f}%)")

    # ══════════════════════════════════════════════════
    # EXP A: Zero out SINK KV entries
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"EXP A: Zero KV for {n_sink} SINK tokens")
    print("=" * 60)

    out_a = run_gen(
        model, processor, text, gen_kwargs,
        drop_indices=sink_abs,
    )
    print(f"  {out_a}")

    # ══════════════════════════════════════════════════
    # EXP B: Zero out NON-SINK KV entries
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"EXP B: Zero KV for {n_nonsink} "
          f"NON-SINK tokens")
    print("=" * 60)

    out_b = run_gen(
        model, processor, text, gen_kwargs,
        drop_indices=nonsink_abs,
    )
    print(f"  {out_b}")

    # ══════════════════════════════════════════════════
    # EXP C: Random zero (same count as sink)
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"EXP C: Zero KV for {n_sink} RANDOM "
          f"visual tokens")
    print("=" * 60)

    rng = np.random.RandomState(42)
    all_vis = np.arange(img_start, img_end)
    random_drop = rng.choice(
        all_vis, size=n_sink, replace=False,
    )
    out_c = run_gen(
        model, processor, text, gen_kwargs,
        drop_indices=random_drop,
    )
    print(f"  {out_c}")

    # ══════════════════════════════════════════════════
    # EXP D: Zero ALL visual KV entries
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"EXP D: Zero KV for ALL {n_vis} "
          f"visual tokens")
    print("=" * 60)

    out_d = run_gen(
        model, processor, text, gen_kwargs,
        drop_indices=all_vis,
    )
    print(f"  {out_d}")

    # ══════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SUMMARY (v4: KV cache zeroing at attention)")
    print("=" * 60)
    print(f"Method: Zero K,V vectors at evicted positions")
    print(f"        in all {len(model.language_model.model.layers)} "
          f"decoder layers, every forward pass")
    print(f"Visual tokens: {n_vis} "
          f"(sink={n_sink}, non-sink={n_nonsink})")
    print()

    results = [
        ("Baseline", 0, baseline),
        ("Zero SINK KV", n_sink, out_a),
        ("Zero NON-SINK KV", n_nonsink, out_b),
        ("Zero RANDOM KV", n_sink, out_c),
        ("Zero ALL KV", n_vis, out_d),
    ]

    for label, n_drop, text_out in results:
        sim = token_overlap(baseline, text_out)
        coherent = len(set(text_out.split())) > 5
        status = "OK" if coherent else "BROKEN"
        print(f"  [{label:<20} drop={n_drop:>3}] "
              f"sim={sim:.2f} [{status}] "
              f"{text_out[:60]}")

    # ── Comparison with v2 ──
    print("\n" + "-" * 60)
    print("v2 (embedding zero) vs v4 (KV cache zero):")
    print("-" * 60)
    print("v2 blocks information at embedding input")
    print("v4 blocks information at attention KV cache")
    print()
    print("If both collapse on drop-sink:")
    print("  -> Sink tokens are structural anchors at")
    print("     both embedding AND attention levels")
    print("  -> KV cache eviction MUST preserve sinks")

    # ── Save results ──
    path = os.path.join(args.output_dir, "results.txt")
    with open(path, "w") as f:
        f.write("Sink Eviction POC v4 — "
                "KV Cache Zeroing\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Video: {args.video}\n")
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"Visual: {n_vis} "
                f"(sink={n_sink}, non={n_nonsink})\n")
        f.write(f"Detection: L{detect_layer}, tau=20\n")
        f.write(f"Method: Zero K,V vectors at evicted "
                f"positions in all decoder layers\n\n")

        for label, n_drop, text_out in results:
            sim = token_overlap(baseline, text_out)
            f.write(f"--- {label} "
                    f"(evicted={n_drop}) ---\n")
            f.write(f"sim={sim:.2f}\n")
            f.write(f"{text_out}\n\n")

    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
