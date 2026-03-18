"""
POC v2: Sink-Aware Token Eviction for Video VLMs.

Key fix from v1: The attention_mask in mlx-vlm is ONLY used
for RoPE position calculation, NOT for actual attention masking.
Setting mask=0 does nothing to prevent attention.

New approach: Instead of masking, we REPLACE the embeddings of
evicted tokens with zeros (or a neutral embedding). This truly
removes visual information from those positions.

Usage:
  uv run --with "transformers<4.52" --with torch \
    --with matplotlib \
    python sink_eviction_poc_v2.py \
      --video test_video.mp4 \
      --output-dir sink_eviction_output_v2
"""

import argparse
import os
from typing import List, Tuple

import mlx.core as mx
import numpy as np

from sink_detect import (
    CaptureStore,
    find_image_token_range,
    patch_model,
    rmsnorm,
)


def detect_sink_indices(
    hidden_states: List[np.ndarray],
    sink_dims: List[int],
    img_start: int,
    img_end: int,
    tau: float = 20.0,
    layer: int = 14,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect sink/non-sink visual token indices."""
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


# ── Monkey-patch to intercept embeddings ─────────────

_original_get_input_embeddings = None
_zero_out_indices = None


def _patched_get_input_embeddings(
    self, input_ids=None, pixel_values=None, **kwargs
):
    """
    Intercept get_input_embeddings to zero out specific
    token positions AFTER vision encoding.
    """
    result = _original_get_input_embeddings(
        self, input_ids, pixel_values, **kwargs,
    )

    if _zero_out_indices is not None and len(
        _zero_out_indices
    ) > 0:
        embeds = result.inputs_embeds
        # Zero out the specified positions
        # Create a mask: 1 everywhere, 0 at drop positions
        seq_len = embeds.shape[1]
        keep = np.ones((1, seq_len, 1), dtype=np.float32)
        for idx in _zero_out_indices:
            if idx < seq_len:
                keep[0, idx, 0] = 0.0
        keep_mx = mx.array(keep)
        new_embeds = embeds * keep_mx

        # Replace the embeddings in the result
        result.inputs_embeds = new_embeds

    return result


def run_with_eviction(
    model,
    processor,
    prompt_text: str,
    extra_kwargs: dict,
    drop_indices: np.ndarray,
    max_tokens: int = 100,
) -> str:
    """Run generate with specific tokens zeroed out."""
    global _zero_out_indices
    from mlx_vlm import generate

    _zero_out_indices = drop_indices

    output = generate(
        model, processor, prompt=prompt_text,
        verbose=False, **extra_kwargs,
    )

    _zero_out_indices = None
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
        "--output-dir", default="sink_eviction_output_v2",
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
    patch_model(model)

    # Patch get_input_embeddings for eviction
    global _original_get_input_embeddings
    import types
    _original_get_input_embeddings = (
        model.get_input_embeddings.__func__
    )

    def _bound_patch(self, input_ids=None,
                     pixel_values=None, **kwargs):
        return _patched_get_input_embeddings(
            self, input_ids, pixel_values, **kwargs,
        )

    model.get_input_embeddings = types.MethodType(
        _bound_patch, model,
    )

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

    # Common kwargs for all experiments
    gen_kwargs = dict(extra_kwargs)
    gen_kwargs["video"] = [args.video]
    gen_kwargs["input_ids"] = input_ids
    gen_kwargs["pixel_values"] = pixel_values
    gen_kwargs["mask"] = mask_input
    gen_kwargs["temperature"] = 0.0
    gen_kwargs["max_tokens"] = args.max_tokens

    # ── Baseline: capture hidden states ──
    print("\n" + "=" * 60)
    print("BASELINE (full embeddings)")
    print("=" * 60)

    CaptureStore.enable()
    baseline = run_with_eviction(
        model, processor, text, gen_kwargs,
        drop_indices=np.array([]),
    )
    CaptureStore.disable()
    print(f"→ {baseline}")

    # Detect sinks from captured hidden states
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
    print(f"  Sink: {n_sink} ({100*n_sink/n_vis:.0f}%)")
    print(f"  Non-sink: {n_nonsink} "
          f"({100*n_nonsink/n_vis:.0f}%)")

    # ── Exp A: Zero out SINK embeddings ──
    print("\n" + "=" * 60)
    print(f"EXP A: Zero out {n_sink} SINK tokens")
    print("=" * 60)
    out_a = run_with_eviction(
        model, processor, text, gen_kwargs,
        drop_indices=sink_abs,
    )
    print(f"→ {out_a}")

    # ── Exp B: Zero out NON-SINK embeddings ──
    print("\n" + "=" * 60)
    print(f"EXP B: Zero out {n_nonsink} NON-SINK tokens")
    print("=" * 60)
    out_b = run_with_eviction(
        model, processor, text, gen_kwargs,
        drop_indices=nonsink_abs,
    )
    print(f"→ {out_b}")

    # ── Exp C: Random zero out (same count as sink) ──
    print("\n" + "=" * 60)
    print(f"EXP C: Random zero out {n_sink} tokens")
    print("=" * 60)
    rng = np.random.RandomState(42)
    all_vis = np.arange(img_start, img_end)
    random_drop = rng.choice(
        all_vis, size=n_sink, replace=False,
    )
    out_c = run_with_eviction(
        model, processor, text, gen_kwargs,
        drop_indices=random_drop,
    )
    print(f"→ {out_c}")

    # ── Exp D: Zero out ALL visual embeddings ──
    print("\n" + "=" * 60)
    print(f"EXP D: Zero out ALL {n_vis} visual tokens")
    print("=" * 60)
    out_d = run_with_eviction(
        model, processor, text, gen_kwargs,
        drop_indices=all_vis,
    )
    print(f"→ {out_d}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY (v2: embedding zeroing)")
    print("=" * 60)
    print(f"Visual tokens: {n_vis} "
          f"(sink={n_sink}, non-sink={n_nonsink})")
    print()

    results = [
        ("Baseline", 0, baseline),
        ("Drop SINK", n_sink, out_a),
        ("Drop NON-SINK", n_nonsink, out_b),
        ("Random drop", n_sink, out_c),
        ("Drop ALL vis", n_vis, out_d),
    ]

    for label, n_drop, text_out in results:
        print(f"[{label:<15} drop={n_drop:>3}] "
              f"{text_out[:70]}")

    # ── Save ──
    path = os.path.join(args.output_dir, "results.txt")
    with open(path, "w") as f:
        f.write("Sink-Aware Eviction POC v2 "
                "(embedding zeroing)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Video: {args.video}\n")
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"Visual: {n_vis} "
                f"(sink={n_sink}, non={n_nonsink})\n")
        f.write(f"Detection: L{detect_layer}, τ=20\n\n")
        for label, n_drop, text_out in results:
            f.write(f"--- {label} (drop={n_drop}) ---\n")
            f.write(f"{text_out}\n\n")
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
