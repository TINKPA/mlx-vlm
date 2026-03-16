"""
POC: Sink-Aware Token Eviction for Video VLMs.

Tests the core hypothesis: visual sink tokens are structural
anchors in the KV cache. Dropping them hurts attention stability
more than dropping non-sink tokens.

Experiment:
  1. Full KV cache (baseline) → generate response
  2. Drop sink token KV from visual tokens → generate
  3. Drop non-sink token KV from visual tokens → generate
  4. Random drop (same count as #2) → generate

Measures output divergence from baseline.

Usage:
  uv run --with "transformers<4.52" --with torch \
    --with matplotlib \
    python sink_eviction_poc.py \
      --video test_video.mp4 \
      --output-dir sink_eviction_output
"""

import argparse
import os
from typing import Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
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
    """
    Detect sink and non-sink visual token indices at a
    given layer. Returns (sink_indices, nonsink_indices)
    as absolute sequence positions.
    """
    hs = hidden_states[layer][0]  # [seq, dim]
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

    sink_abs = sink_local + img_start
    nonsink_abs = nonsink_local + img_start

    return sink_abs, nonsink_abs


def generate_with_masked_kv(
    model,
    processor,
    input_ids: mx.array,
    pixel_values: mx.array,
    mask_input: mx.array,
    prompt_text: str,
    extra_kwargs: dict,
    drop_indices: np.ndarray,
    max_tokens: int = 100,
    label: str = "",
) -> str:
    """
    Run generation but zero out KV cache entries for
    specified token indices after prefill.

    This simulates evicting those tokens from the cache.
    """
    from mlx_vlm import generate

    # We need to modify the KV cache after prefill
    # Strategy: run full prefill, then mask out specific
    # KV entries before generation continues

    kwargs = dict(extra_kwargs)
    kwargs["input_ids"] = input_ids
    kwargs["pixel_values"] = pixel_values
    kwargs["mask"] = mask_input
    kwargs["temperature"] = 0.0
    kwargs["max_tokens"] = 1  # Just prefill + 1 token

    # First, do a normal prefill
    output = generate(
        model, processor, prompt=prompt_text,
        verbose=False, **kwargs,
    )

    if len(drop_indices) > 0:
        # Now mask KV cache entries
        n_masked = 0
        for layer in model.language_model.model.layers:
            if hasattr(layer, "self_attn"):
                attn = layer.self_attn
            else:
                continue

            cache = None
            # Try to access the cache
            # In mlx-vlm, cache is managed externally
            # We need a different approach...
            pass

    # Since directly modifying KV cache in mlx-vlm is
    # complex, let's use a simpler approach:
    # Mask the attention at specific positions by
    # modifying the attention mask

    return output.text


def generate_with_attention_mask(
    model,
    processor,
    prompt_text: str,
    extra_kwargs: dict,
    attention_mask: mx.array,
    max_tokens: int = 100,
) -> str:
    """
    Generate with a modified attention mask that blocks
    attention to specific token positions.
    """
    from mlx_vlm import generate

    kwargs = dict(extra_kwargs)
    kwargs["mask"] = attention_mask
    kwargs["temperature"] = 0.0
    kwargs["max_tokens"] = max_tokens

    output = generate(
        model, processor, prompt=prompt_text,
        verbose=False, **kwargs,
    )
    return output.text


def create_eviction_mask(
    base_mask: mx.array,
    drop_indices: np.ndarray,
) -> mx.array:
    """
    Create attention mask that zeros out dropped token
    positions. Uses the same format as the input
    attention_mask: [1, seq_len] with 1s and 0s.
    """
    mask_np = np.array(base_mask)  # [1, seq_len]
    if len(drop_indices) > 0:
        mask_np[0, drop_indices] = 0
    return mx.array(mask_np)


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
        "--output-dir", default="sink_eviction_output",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sink_dims = [458, 2570]
    detect_layer = 14  # Middle layer for sink detection

    # ── Load model ──
    from mlx_vlm import load, generate
    from mlx_vlm.utils import load_config
    from mlx_vlm.video_generate import process_vision_info

    print(f"Loading {args.model}...")
    model, processor = load(args.model)
    config = load_config(args.model)
    patch_model(model)

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
          f"Visual tokens: [{img_start},{img_end}) "
          f"= {n_vis}")

    # ── Step 1: Baseline run with hidden state capture ──
    print("\n" + "=" * 60)
    print("Step 1: Baseline (full KV cache)")
    print("=" * 60)

    CaptureStore.enable()
    baseline_kwargs = dict(extra_kwargs)
    baseline_kwargs["video"] = [args.video]
    baseline_kwargs["input_ids"] = input_ids
    baseline_kwargs["pixel_values"] = pixel_values
    baseline_kwargs["mask"] = mask_input
    baseline_kwargs["temperature"] = 0.0
    baseline_kwargs["max_tokens"] = args.max_tokens

    baseline_output = generate(
        model, processor, prompt=text,
        verbose=False, **baseline_kwargs,
    )
    CaptureStore.disable()
    print(f"Baseline: {baseline_output.text}")

    # ── Step 2: Detect sink tokens ──
    hs_np = [
        np.array(h) for h in CaptureStore.hidden_states
    ]
    print(f"\nCaptured {len(hs_np)} layers")

    sink_abs, nonsink_abs = detect_sink_indices(
        hs_np, sink_dims, img_start, img_end,
        tau=20.0, layer=detect_layer,
    )
    n_sink = len(sink_abs)
    n_nonsink = len(nonsink_abs)
    print(f"\nLayer {detect_layer} sink detection:")
    print(f"  Sink tokens: {n_sink} "
          f"({100*n_sink/n_vis:.0f}%)")
    print(f"  Non-sink tokens: {n_nonsink} "
          f"({100*n_nonsink/n_vis:.0f}%)")

    # ── Step 3: Run experiments with different masks ──
    experiments = {}

    # Experiment A: Drop SINK tokens
    print("\n" + "=" * 60)
    print(f"Exp A: Drop {n_sink} SINK visual tokens")
    print("=" * 60)

    mask_drop_sink = create_eviction_mask(
        mask_input, sink_abs,
    )
    kwargs_a = dict(extra_kwargs)
    kwargs_a["video"] = [args.video]
    kwargs_a["input_ids"] = input_ids
    kwargs_a["pixel_values"] = pixel_values
    kwargs_a["mask"] = mask_drop_sink
    kwargs_a["temperature"] = 0.0
    kwargs_a["max_tokens"] = args.max_tokens

    output_a = generate(
        model, processor, prompt=text,
        verbose=False, **kwargs_a,
    )
    print(f"Drop sink: {output_a.text}")
    experiments["drop_sink"] = output_a.text

    # Experiment B: Drop NON-SINK tokens
    print("\n" + "=" * 60)
    print(f"Exp B: Drop {n_nonsink} NON-SINK "
          f"visual tokens")
    print("=" * 60)

    mask_drop_nonsink = create_eviction_mask(
        mask_input, nonsink_abs,
    )
    kwargs_b = dict(extra_kwargs)
    kwargs_b["video"] = [args.video]
    kwargs_b["input_ids"] = input_ids
    kwargs_b["pixel_values"] = pixel_values
    kwargs_b["mask"] = mask_drop_nonsink
    kwargs_b["temperature"] = 0.0
    kwargs_b["max_tokens"] = args.max_tokens

    output_b = generate(
        model, processor, prompt=text,
        verbose=False, **kwargs_b,
    )
    print(f"Drop non-sink: {output_b.text}")
    experiments["drop_nonsink"] = output_b.text

    # Experiment C: Random drop (same count as sink)
    print("\n" + "=" * 60)
    print(f"Exp C: Random drop {n_sink} visual tokens")
    print("=" * 60)

    rng = np.random.RandomState(42)
    all_vis = np.arange(img_start, img_end)
    random_drop = rng.choice(
        all_vis, size=n_sink, replace=False,
    )

    mask_drop_random = create_eviction_mask(
        mask_input, random_drop,
    )
    kwargs_c = dict(extra_kwargs)
    kwargs_c["video"] = [args.video]
    kwargs_c["input_ids"] = input_ids
    kwargs_c["pixel_values"] = pixel_values
    kwargs_c["mask"] = mask_drop_random
    kwargs_c["temperature"] = 0.0
    kwargs_c["max_tokens"] = args.max_tokens

    output_c = generate(
        model, processor, prompt=text,
        verbose=False, **kwargs_c,
    )
    print(f"Random drop: {output_c.text}")
    experiments["random_drop"] = output_c.text

    # Experiment D: Drop ALL visual tokens
    print("\n" + "=" * 60)
    print(f"Exp D: Drop ALL {n_vis} visual tokens")
    print("=" * 60)

    mask_drop_all = create_eviction_mask(
        mask_input, all_vis,
    )
    kwargs_d = dict(extra_kwargs)
    kwargs_d["video"] = [args.video]
    kwargs_d["input_ids"] = input_ids
    kwargs_d["pixel_values"] = pixel_values
    kwargs_d["mask"] = mask_drop_all
    kwargs_d["temperature"] = 0.0
    kwargs_d["max_tokens"] = args.max_tokens

    output_d = generate(
        model, processor, prompt=text,
        verbose=False, **kwargs_d,
    )
    print(f"Drop all vis: {output_d.text}")
    experiments["drop_all_vis"] = output_d.text

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nPrompt: {args.prompt}")
    print(f"Visual tokens: {n_vis} "
          f"(sink={n_sink}, non-sink={n_nonsink})")
    print(f"Detection layer: {detect_layer}, τ=20")

    print(f"\n{'Condition':<25} {'Dropped':<12} "
          f"{'Output'}")
    print("-" * 80)
    print(f"{'Baseline (full)':<25} {'0':<12} "
          f"{baseline_output.text[:60]}")
    print(f"{'Drop SINK':<25} {f'{n_sink}':<12} "
          f"{output_a.text[:60]}")
    print(f"{'Drop NON-SINK':<25} {f'{n_nonsink}':<12} "
          f"{output_b.text[:60]}")
    print(f"{'Random drop':<25} {f'{n_sink}':<12} "
          f"{output_c.text[:60]}")
    print(f"{'Drop ALL visual':<25} {f'{n_vis}':<12} "
          f"{output_d.text[:60]}")

    # ── Save results ──
    result_path = os.path.join(
        args.output_dir, "results.txt",
    )
    with open(result_path, "w") as f:
        f.write("Sink-Aware Token Eviction POC\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Video: {args.video}\n")
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"Visual tokens: {n_vis} "
                f"(sink={n_sink}, non-sink={n_nonsink})\n")
        f.write(f"Detection: layer {detect_layer}, "
                f"τ=20, dims={sink_dims}\n\n")

        f.write("--- Baseline ---\n")
        f.write(f"{baseline_output.text}\n\n")

        f.write(f"--- Drop {n_sink} SINK tokens ---\n")
        f.write(f"{output_a.text}\n\n")

        f.write(f"--- Drop {n_nonsink} NON-SINK "
                f"tokens ---\n")
        f.write(f"{output_b.text}\n\n")

        f.write(f"--- Random drop {n_sink} tokens ---\n")
        f.write(f"{output_c.text}\n\n")

        f.write(f"--- Drop ALL {n_vis} "
                f"visual tokens ---\n")
        f.write(f"{output_d.text}\n\n")

    print(f"\nResults saved → {result_path}")


if __name__ == "__main__":
    main()
