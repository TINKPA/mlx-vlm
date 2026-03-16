"""
Compare Visual Attention Sink across:
  - Qwen2-VL vs Qwen2.5-VL (model comparison)
  - Video vs single image (modality comparison)

Outputs a 2x2 grid: rows=model, cols=input_type.

Usage:
  uv run --with "transformers<4.52" --with torch \
    --with seaborn --with matplotlib \
    python sink_compare.py \
      --video test_video.mp4 \
      --image test_single_frame.jpg \
      --output-dir sink_compare_output
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import cv2
import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ── Capture store ─────────────────────────────────────────────

class CaptureStore:
    hidden_states: List[mx.array] = []
    attn_weights: List[mx.array] = []
    enabled: bool = False

    @classmethod
    def reset(cls):
        cls.hidden_states = []
        cls.attn_weights = []

    @classmethod
    def enable(cls):
        cls.enabled = True
        cls.reset()

    @classmethod
    def disable(cls):
        cls.enabled = False


# ── Patching (works for both qwen2_vl and qwen2_5_vl) ────────

_original_decoder_call = None


def _patched_decoder_call(self, x, mask=None, cache=None,
                          position_ids=None):
    if CaptureStore.enabled and x.shape[1] > 1:
        CaptureStore.hidden_states.append(x)
    return _original_decoder_call(
        self, x, mask=mask, cache=cache,
        position_ids=position_ids,
    )


def patch_for_model(model, model_id: str):
    """Patch the correct decoder layer class."""
    global _original_decoder_call

    if "qwen2_5" in model_id.replace("-", "_").replace(
        ".", "_"
    ).lower() or "Qwen2.5" in model_id:
        from mlx_vlm.models.qwen2_5_vl.language import (
            Qwen2VLDecoderLayer,
        )
    else:
        from mlx_vlm.models.qwen2_vl.language import (
            Qwen2VLDecoderLayer,
        )

    _original_decoder_call = Qwen2VLDecoderLayer.__call__
    Qwen2VLDecoderLayer.__call__ = _patched_decoder_call
    n = len(model.language_model.model.layers)
    print(f"  Patched {n} decoder layers.")


def unpatch_model(model_id: str):
    """Restore original decoder call."""
    global _original_decoder_call
    if _original_decoder_call is None:
        return
    if "qwen2_5" in model_id.replace("-", "_").replace(
        ".", "_"
    ).lower() or "Qwen2.5" in model_id:
        from mlx_vlm.models.qwen2_5_vl.language import (
            Qwen2VLDecoderLayer,
        )
    else:
        from mlx_vlm.models.qwen2_vl.language import (
            Qwen2VLDecoderLayer,
        )
    Qwen2VLDecoderLayer.__call__ = _original_decoder_call
    _original_decoder_call = None


# ── Utilities ─────────────────────────────────────────────────

def find_image_token_range(input_ids, config):
    ids = input_ids.flatten().tolist()
    img_id = config.image_token_id
    vid_id = config.video_token_id
    token_id = img_id if img_id in ids else vid_id
    positions = [i for i, t in enumerate(ids) if t == token_id]
    if not positions:
        raise ValueError("No image/video tokens found.")
    return positions[0], positions[-1] + 1


def rmsnorm(hs, eps=1e-6):
    variance = np.mean(hs ** 2, axis=-1, keepdims=True)
    return hs * np.reciprocal(np.sqrt(variance + eps))


def compute_phi_per_layer(
    hidden_states: List[np.ndarray],
    sink_dims: List[int],
    img_start: int,
    img_end: int,
) -> Dict:
    """Compute phi stats per layer for visual tokens."""
    results = {}
    n_layers = len(hidden_states)
    for li in range(2, n_layers):
        hs = hidden_states[li][0]
        rms = np.abs(rmsnorm(hs))
        phi = np.max(
            np.stack([rms[:, d] for d in sink_dims], axis=-1),
            axis=-1,
        )
        vis_phi = phi[img_start:img_end]
        bos_phi = float(phi[0])
        n_sink_20 = int((vis_phi > 20).sum())
        n_total = len(vis_phi)
        results[li] = {
            "vis_phi": vis_phi,
            "bos_phi": bos_phi,
            "n_sink_20": n_sink_20,
            "n_total": n_total,
            "pct_20": 100 * n_sink_20 / max(n_total, 1),
            "mean_phi": float(vis_phi.mean()),
            "median_phi": float(np.median(vis_phi)),
        }
    return results


# ── Run one experiment ────────────────────────────────────────

def run_experiment(
    model_id: str,
    input_type: str,
    video_path: str,
    image_path: str,
    prompt: str,
    max_tokens: int,
    fps: float,
):
    """
    Load model, run inference, capture hidden states,
    compute phi. Returns phi_results dict.
    """
    from mlx_vlm import load, generate
    from mlx_vlm.utils import load_config

    print(f"\n{'='*50}")
    print(f"Experiment: {model_id} + {input_type}")
    print(f"{'='*50}")

    print(f"  Loading {model_id}...")
    model, processor = load(model_id)
    config = load_config(model_id)
    patch_for_model(model, model_id)

    model_config = model.config
    sink_dims = [458, 2570]

    if input_type == "video":
        from mlx_vlm.video_generate import (
            process_vision_info,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 224 * 224,
                        "fps": fps,
                    },
                    {"type": "text", "text": prompt},
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
    else:
        # Single image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True,
        )
        from mlx_vlm.video_generate import (
            process_vision_info,
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
    if pv is None:
        raise ValueError("No pixel values.")
    pixel_values = mx.array(pv)
    mask_input = mx.array(inputs["attention_mask"])

    kwargs = {}
    for key in [
        "video_grid_thw", "image_grid_thw",
    ]:
        if inputs.get(key) is not None:
            kwargs[key] = mx.array(inputs[key])

    img_start, img_end = find_image_token_range(
        input_ids, model_config,
    )
    n_vis = img_end - img_start
    print(f"  Visual tokens: {n_vis}, "
          f"seq len: {input_ids.shape[1]}")

    # Run
    CaptureStore.enable()
    gen_kwargs = dict(kwargs)
    if input_type == "video":
        gen_kwargs["video"] = [video_path]
    gen_kwargs["input_ids"] = input_ids
    gen_kwargs["pixel_values"] = pixel_values
    gen_kwargs["mask"] = mask_input
    gen_kwargs["temperature"] = 0.0
    gen_kwargs["max_tokens"] = max_tokens

    output = generate(
        model, processor, prompt=text,
        verbose=False, **gen_kwargs,
    )
    print(f"  Output: {output.text[:60]}...")
    CaptureStore.disable()

    # Convert
    hs_np = [np.array(h) for h in CaptureStore.hidden_states]
    n_layers = len(hs_np)
    hidden_dim = hs_np[0].shape[-1] if hs_np else 0
    print(f"  Captured {n_layers} layers, dim={hidden_dim}")

    # Compute phi
    phi_results = compute_phi_per_layer(
        hs_np, sink_dims, img_start, img_end,
    )

    # Cleanup
    unpatch_model(model_id)
    del model, processor
    import gc
    gc.collect()

    return phi_results, n_vis, hidden_dim


# ── Plotting ──────────────────────────────────────────────────

def plot_comparison(
    all_results: Dict,
    output_dir: str,
):
    """
    Plot 2x2 comparison grid.
    Rows: Qwen2-VL, Qwen2.5-VL
    Cols: Image, Video
    Each cell: phi distribution + sink% per layer
    """
    import matplotlib.pyplot as plt

    models = list(all_results.keys())
    input_types = ["image", "video"]

    # ── Plot 1: Sink percentage per layer ──
    fig, axes = plt.subplots(
        len(models), len(input_types),
        figsize=(14, 5 * len(models)),
        sharey=True,
    )
    if len(models) == 1:
        axes = axes.reshape(1, -1)

    for row, model_name in enumerate(models):
        for col, itype in enumerate(input_types):
            key = (model_name, itype)
            if key not in all_results[model_name]:
                axes[row, col].text(
                    0.5, 0.5, "N/A",
                    ha="center", va="center",
                )
                continue

            phi_res = all_results[model_name][itype]
            layers = sorted(phi_res.keys())
            pcts = [phi_res[l]["pct_20"] for l in layers]
            means = [phi_res[l]["mean_phi"] for l in layers]
            n_vis = phi_res[layers[0]]["n_total"]

            ax = axes[row, col]
            ax.bar(
                layers, pcts, color="coral",
                alpha=0.7, label="sink% (τ=20)",
            )
            ax2 = ax.twinx()
            ax2.plot(
                layers, means, "b-o",
                markersize=3, label="mean φ",
            )
            ax2.axhline(
                y=20, color="green", linestyle="--",
                alpha=0.5,
            )
            ax2.set_ylabel("mean φ", color="blue")

            short_name = model_name.split("/")[-1][:25]
            ax.set_title(
                f"{short_name} + {itype}\n"
                f"({n_vis} vis tokens)",
                fontsize=11,
            )
            ax.set_xlabel("Layer")
            ax.set_ylabel("% Sink (τ=20)")
            ax.set_ylim(0, 105)

            if row == 0 and col == 0:
                ax.legend(loc="upper left", fontsize=8)
                ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Visual Attention Sink: "
        "Model × Input Type Comparison\n"
        "sink dims = {458, 2570}, τ = 20",
        fontsize=14,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "sink_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")

    # ── Plot 2: Phi histograms at key layers ──
    key_layers = [2, 10, 20]
    fig, axes = plt.subplots(
        len(key_layers),
        len(models) * len(input_types),
        figsize=(4 * len(models) * len(input_types),
                 3 * len(key_layers)),
    )

    col_idx = 0
    for model_name in models:
        for itype in input_types:
            if itype not in all_results[model_name]:
                col_idx += 1
                continue
            phi_res = all_results[model_name][itype]
            for row, li in enumerate(key_layers):
                if li not in phi_res:
                    # Find closest
                    li = min(
                        phi_res.keys(),
                        key=lambda x: abs(x - li),
                    )
                vis_phi = phi_res[li]["vis_phi"]
                ax = axes[row, col_idx]
                ax.hist(
                    vis_phi, bins=40,
                    color="steelblue", alpha=0.7,
                    edgecolor="white",
                )
                ax.axvline(
                    x=20, color="red",
                    linestyle="--", linewidth=2,
                )
                pct = phi_res[li]["pct_20"]
                short = model_name.split("/")[-1][:20]
                ax.set_title(
                    f"L{li}: {short}+{itype}\n"
                    f"sink={pct:.0f}%",
                    fontsize=9,
                )
                if row == len(key_layers) - 1:
                    ax.set_xlabel("φ")
                if col_idx == 0:
                    ax.set_ylabel("Count")
            col_idx += 1

    fig.suptitle(
        "φ Distributions at Key Layers\n"
        "Red line = τ=20",
        fontsize=14,
    )
    plt.tight_layout()
    path = os.path.join(
        output_dir, "phi_histograms.png",
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")

    # ── Print summary table ──
    print("\n" + "=" * 60)
    print("SUMMARY: Sink % at τ=20 (selected layers)")
    print("=" * 60)
    header = f"{'Model':<30} {'Input':<7}"
    for li in [2, 5, 10, 15, 20, 25]:
        header += f" L{li:>2}"
    print(header)
    print("-" * 60)

    for model_name in models:
        for itype in input_types:
            if itype not in all_results[model_name]:
                continue
            phi_res = all_results[model_name][itype]
            short = model_name.split("/")[-1][:28]
            line = f"{short:<30} {itype:<7}"
            for li in [2, 5, 10, 15, 20, 25]:
                if li in phi_res:
                    line += f" {phi_res[li]['pct_20']:>3.0f}"
                else:
                    line += "  --"
            print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--prompt",
        default="Describe what you see in detail.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=50,
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument(
        "--output-dir", default="sink_compare_output",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    models = [
        "mlx-community/Qwen2-VL-7B-Instruct-4bit",
        "mlx-community/Qwen2.5-VL-7B-Instruct-8bit",
    ]

    all_results = {}

    for model_id in models:
        all_results[model_id] = {}
        for itype in ["image", "video"]:
            try:
                phi_res, n_vis, dim = run_experiment(
                    model_id, itype,
                    args.video, args.image,
                    args.prompt, args.max_tokens,
                    args.fps,
                )
                all_results[model_id][itype] = phi_res
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()

    # ── Plot ──
    print("\n" + "=" * 50)
    print("Generating comparison plots...")
    print("=" * 50)
    plot_comparison(all_results, args.output_dir)

    print(f"\nAll outputs → {args.output_dir}/")


if __name__ == "__main__":
    main()
