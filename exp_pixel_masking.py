"""
Pixel Masking Ablation: Is sink content-driven or position-driven?

Protocol:
  1. Run normal inference on video, capture hidden states
  2. Detect sink tokens via φ metric, get spatial positions
  3. Zero out sink patches in pixel_values tensor
  4. Re-run inference on masked input
  5. Compare: (a) answer, (b) whether same positions are still sink

Key question answered:
  - If masked positions are NO LONGER sink → content-driven
  - If masked positions are STILL sink → position/structural

Usage:
  uv run --with "transformers<4.52" --with torch \
    --with matplotlib \
    python exp_pixel_masking.py \
      --data-dir /Volumes/RAID0/datasets/MVBench \
      --tasks action_antonym \
      --max-questions 5 \
      --output-dir exp_pixel_masking_output
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from sink_detect import (
    CaptureStore,
    find_image_token_range,
    patch_model,
    rmsnorm,
)
from sink_eviction_v10_mvbench import (
    load_mvbench_tasks,
    find_video,
    format_mcq_prompt,
    extract_answer_mvbench,
)


# ── Sink detection on captured hidden states ──────────

def detect_sink_positions(
    hidden_states: list,
    img_start: int,
    img_end: int,
    sink_dims: List[int],
    layer: int = 14,
    tau: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect sink tokens and return their local indices
    within the visual token range + phi values.

    Returns:
        sink_local_indices: indices relative to img_start
        vis_phi: phi values for all visual tokens
    """
    hs = hidden_states[layer]
    if isinstance(hs, mx.array):
        hs = np.array(hs)
    if hs.ndim == 3:
        hs = hs[0]  # [seq_len, hidden_dim]

    rms_val = np.abs(rmsnorm(hs))
    phi = np.max(
        np.stack([rms_val[:, d] for d in sink_dims], axis=-1),
        axis=-1,
    )
    vis_phi = phi[img_start:img_end]
    sink_mask = vis_phi > tau
    sink_local = np.where(sink_mask)[0]
    return sink_local, vis_phi


def sink_indices_to_pixel_mask(
    sink_local_indices: np.ndarray,
    grid_thw: Tuple[int, int, int],
    spatial_merge_size: int,
) -> np.ndarray:
    """
    Map sink token indices (in LLM space) back to patch
    indices in the pixel_values tensor.

    Qwen2.5-VL layout:
      - Vision encoder outputs patches at grid (T, H, W)
      - spatial_merge merges spatial_merge_size^2 patches
        into 1 LLM token
      - So LLM grid is (T, H//sms, W//sms)
      - LLM token i maps to sms^2 patches in the encoder

    Returns: array of patch indices to zero out in pixel_values
    """
    T, H, W = grid_thw
    sms = spatial_merge_size
    llm_H = H // sms
    llm_W = W // sms
    tokens_per_frame = llm_H * llm_W

    patch_indices = []
    patches_per_frame = H * W

    for idx in sink_local_indices:
        # Which frame and spatial position in LLM grid
        frame = idx // tokens_per_frame
        local = idx % tokens_per_frame
        row = local // llm_W
        col = local % llm_W

        # Map to encoder patch positions
        for dr in range(sms):
            for dc in range(sms):
                pr = row * sms + dr
                pc = col * sms + dc
                patch_idx = frame * patches_per_frame + pr * W + pc
                patch_indices.append(patch_idx)

    return np.array(patch_indices, dtype=np.int64)


# ── Full prefill with capture ─────────────────────────

def run_with_capture(
    model, processor, ids, pv, mask, extra,
    max_tok=30,
):
    """
    Run full prefill + generate, capturing hidden states.
    Returns (output_text, hidden_states_np).
    """
    from mlx_vlm.models.cache import make_prompt_cache

    CaptureStore.enable()

    cache = make_prompt_cache(model.language_model)
    eo = model.get_input_embeddings(
        ids, pv, mask=mask, **extra,
    )
    embeds = eo.inputs_embeds
    fkw = {
        k: v for k, v in eo.to_dict().items()
        if k != "inputs_embeds" and v is not None
    }
    out = model.language_model(
        ids, inputs_embeds=embeds, cache=cache, **fkw,
    )
    mx.eval([c.state for c in cache])
    logits = out.logits[:, -1, :]

    # Capture hidden states before generating
    hs_np = [np.array(h) for h in CaptureStore.hidden_states]
    CaptureStore.disable()

    # Generate
    from sink_eviction_v9_streaming import generate_from_cache
    text = generate_from_cache(
        model, processor, cache, logits, max_tok,
    )
    return text, hs_np


# ── Main ──────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-VL-7B-Instruct-8bit",
    )
    ap.add_argument(
        "--data-dir",
        default="/Volumes/RAID0/datasets/MVBench",
    )
    ap.add_argument("--max-tokens", type=int, default=30)
    ap.add_argument(
        "--output-dir", default="exp_pixel_masking_output",
    )
    ap.add_argument(
        "--max-questions", type=int, default=5,
    )
    ap.add_argument(
        "--tasks", type=str, nargs="+",
        default=None,
        help="Tasks to run (default=all 20 MVBench tasks)",
    )
    ap.add_argument("--tau", type=float, default=20.0)
    ap.add_argument("--detect-layer", type=int, default=14)
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for question sampling",
    )
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sink_dims = [458, 2570]

    # ── Load data ─────────────────────────────────────
    print(f"Loading MVBench from {args.data_dir}...")
    all_tasks = load_mvbench_tasks(args.data_dir)
    if args.tasks:
        all_tasks = {
            k: v for k, v in all_tasks.items()
            if k in args.tasks
        }

    # Sample questions
    rng = np.random.RandomState(args.seed)
    sampled = {}
    per_task = max(
        1, args.max_questions // len(all_tasks),
    )
    for tname, qs in all_tasks.items():
        n = min(per_task, len(qs))
        idx = rng.choice(len(qs), n, replace=False)
        sampled[tname] = [qs[i] for i in idx]
    total_q = sum(len(v) for v in sampled.values())
    print(f"  {total_q} questions across "
          f"{len(sampled)} tasks")

    # ── Load model ────────────────────────────────────
    from mlx_vlm import load
    from mlx_vlm.video_generate import process_vision_info

    print(f"\nLoading {args.model}...")
    model, processor = load(args.model)
    patch_model(model)
    mcfg = model.config
    sms = mcfg.vision_config.spatial_merge_size

    # ── Run experiments ───────────────────────────────
    results = []
    t0 = time.time()

    for tname in sorted(sampled.keys()):
        questions = sampled[tname]
        print(f"\n--- {tname} ({len(questions)} Qs) ---")

        for qi, item in enumerate(questions):
            video_name = item["video"]
            question = item["question"]
            candidates = item["candidates"]
            gt = item["answer"]

            video_path = find_video(
                video_name, args.data_dir,
            )
            if video_path is None:
                print(f"  SKIP Q{qi}: video not found")
                continue

            try:
                q_start = time.time()

                # Prepare input
                prompt = format_mcq_prompt(
                    question, candidates,
                )
                msgs = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "max_pixels": 224 * 224,
                            "fps": 1.0,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }]
                text = processor.apply_chat_template(
                    msgs, tokenize=False,
                    add_generation_prompt=True,
                )
                img_in, vid_in, _ = process_vision_info(
                    msgs, True,
                )
                inputs = processor(
                    text=[text], images=img_in,
                    videos=vid_in, padding=True,
                    return_tensors="pt",
                )
                ids = mx.array(inputs["input_ids"])
                pvk = (
                    "pixel_values_videos"
                    if "pixel_values_videos" in inputs
                    else "pixel_values"
                )
                pv = mx.array(inputs[pvk])
                attn_mask = mx.array(inputs["attention_mask"])
                extra = {}
                for k in ["video_grid_thw", "image_grid_thw"]:
                    if inputs.get(k) is not None:
                        extra[k] = mx.array(inputs[k])

                s, e = find_image_token_range(ids, mcfg)
                nv = e - s

                # Get grid info
                thw_key = (
                    "video_grid_thw"
                    if "video_grid_thw" in inputs
                    else "image_grid_thw"
                )
                grid_thw = tuple(
                    int(x) for x in inputs[thw_key][0]
                )

                # ── Pass 1: Normal inference ──────────
                out_normal, hs_normal = run_with_capture(
                    model, processor, ids, pv, attn_mask,
                    extra, args.max_tokens,
                )
                pred_normal, _ = extract_answer_mvbench(
                    out_normal, candidates,
                )
                correct_normal = (pred_normal == gt)

                # Detect sinks
                sink_local, vis_phi = detect_sink_positions(
                    hs_normal, s, e, sink_dims,
                    args.detect_layer, args.tau,
                )
                n_sink = len(sink_local)
                sink_frac = n_sink / nv if nv > 0 else 0

                # ── Create masked pixel_values ────────
                patch_idx = sink_indices_to_pixel_mask(
                    sink_local, grid_thw, sms,
                )
                pv_np = np.array(pv)
                n_patches_total = pv_np.shape[1]
                # Clip to valid range
                valid_idx = patch_idx[
                    patch_idx < n_patches_total
                ]
                pv_masked = pv_np.copy()
                pv_masked[0, valid_idx] = 0.0  # zero out
                pv_masked_mx = mx.array(pv_masked)

                n_masked = len(valid_idx)
                mask_frac = (
                    n_masked / n_patches_total
                    if n_patches_total > 0 else 0
                )

                # ── Pass 2: Masked inference ──────────
                out_masked, hs_masked = run_with_capture(
                    model, processor, ids, pv_masked_mx,
                    attn_mask, extra, args.max_tokens,
                )
                pred_masked, _ = extract_answer_mvbench(
                    out_masked, candidates,
                )
                correct_masked = (pred_masked == gt)

                # Detect sinks in masked run
                sink_local_m, vis_phi_m = detect_sink_positions(
                    hs_masked, s, e, sink_dims,
                    args.detect_layer, args.tau,
                )
                n_sink_m = len(sink_local_m)

                # Compare sink positions
                set_orig = set(sink_local.tolist())
                set_masked = set(sink_local_m.tolist())
                overlap = set_orig & set_masked
                iou = (
                    len(overlap)
                    / len(set_orig | set_masked)
                    if (set_orig | set_masked) else 0
                )
                # How many original sinks are still sinks?
                persistence = (
                    len(overlap) / len(set_orig)
                    if set_orig else 0
                )

                q_elapsed = time.time() - q_start

                result = {
                    "task": tname,
                    "qi": qi,
                    "gt": gt,
                    "elapsed_sec": round(q_elapsed, 2),
                    # Normal
                    "pred_normal": pred_normal,
                    "correct_normal": correct_normal,
                    "output_normal": out_normal[:80],
                    # Masked
                    "pred_masked": pred_masked,
                    "correct_masked": correct_masked,
                    "output_masked": out_masked[:80],
                    # Sink stats
                    "n_vis": nv,
                    "n_sink_normal": n_sink,
                    "sink_frac": round(sink_frac, 3),
                    "n_patches_masked": n_masked,
                    "mask_frac": round(mask_frac, 3),
                    # Masked sink stats
                    "n_sink_masked": n_sink_m,
                    "sink_iou": round(iou, 3),
                    "sink_persistence": round(persistence, 3),
                    # Did answer change?
                    "answer_changed": (
                        pred_normal != pred_masked
                    ),
                }
                results.append(result)

                mark_n = "O" if correct_normal else "X"
                mark_m = "O" if correct_masked else "X"
                chg = "CHANGED" if result["answer_changed"] \
                    else "same"
                print(
                    f"  Q{qi}: [{mark_n}]{pred_normal[:20]:20s}"
                    f" → [{mark_m}]{pred_masked[:20]:20s}"
                    f"  {chg}  "
                    f"sink={n_sink}→{n_sink_m} "
                    f"IoU={iou:.2f} "
                    f"persist={persistence:.2f}"
                )

            except Exception as ex:
                print(f"  ERROR Q{qi}: {ex}")
                import traceback
                traceback.print_exc()

    # ── Summary ───────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"PIXEL MASKING ABLATION RESULTS")
    print(f"  {len(results)} questions in "
          f"{elapsed/60:.1f} min")
    print(f"{'='*60}")

    if results:
        n_correct_normal = sum(
            r["correct_normal"] for r in results
        )
        n_correct_masked = sum(
            r["correct_masked"] for r in results
        )
        n_changed = sum(
            r["answer_changed"] for r in results
        )
        avg_iou = np.mean(
            [r["sink_iou"] for r in results]
        )
        avg_persist = np.mean(
            [r["sink_persistence"] for r in results]
        )
        avg_sink_frac = np.mean(
            [r["sink_frac"] for r in results]
        )
        avg_sink_normal = np.mean(
            [r["n_sink_normal"] for r in results]
        )
        avg_sink_masked = np.mean(
            [r["n_sink_masked"] for r in results]
        )

        print(f"\nAccuracy:")
        print(f"  Normal:  {n_correct_normal}/{len(results)}"
              f" ({n_correct_normal/len(results):.1%})")
        print(f"  Masked:  {n_correct_masked}/{len(results)}"
              f" ({n_correct_masked/len(results):.1%})")
        print(f"  Changed: {n_changed}/{len(results)}")

        print(f"\nSink Analysis:")
        print(f"  Avg sink fraction: {avg_sink_frac:.1%}")
        print(f"  Avg sinks (normal): {avg_sink_normal:.1f}")
        print(f"  Avg sinks (masked): {avg_sink_masked:.1f}")
        print(f"  Avg sink IoU: {avg_iou:.3f}")
        print(f"  Avg sink persistence: {avg_persist:.3f}")

        print(f"\nInterpretation:")
        if avg_persist > 0.7:
            print(f"  HIGH persistence ({avg_persist:.2f})"
                  f" → sinks are POSITION-driven")
            print(f"  Blacking out content doesn't prevent "
                  f"sink formation.")
        elif avg_persist > 0.3:
            print(f"  MODERATE persistence ({avg_persist:.2f})"
                  f" → MIXED content + position")
        else:
            print(f"  LOW persistence ({avg_persist:.2f})"
                  f" → sinks are CONTENT-driven")
            print(f"  Blacking out content eliminates sinks.")

    # Save
    jp = os.path.join(args.output_dir, "results.json")
    with open(jp, "w") as f:
        json.dump({
            "experiment": "pixel_masking_ablation",
            "seed": args.seed,
            "detect_layer": args.detect_layer,
            "tau": args.tau,
            "sink_dims": sink_dims,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved → {jp}")


if __name__ == "__main__":
    main()
