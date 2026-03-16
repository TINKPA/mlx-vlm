"""
Adaptive sink analysis: find per-layer tau that separates
sink from non-sink in Qwen2.5-VL on video.

Extends sink_detect.py with:
  - Per-layer phi distribution histograms
  - Adaptive tau via Otsu's method (bimodal threshold)
  - Comparison: fixed tau vs adaptive tau

Usage:
  uv run --with "transformers<4.52" --with torch \
    --with seaborn --with matplotlib \
    python sink_adaptive.py \
      --model mlx-community/Qwen2.5-VL-7B-Instruct-8bit \
      --video test_video.mp4 \
      --output-dir sink_adaptive_output
"""

import argparse
import os
from typing import List, Tuple

import cv2
import mlx.core as mx
import numpy as np

# Reuse capture infrastructure from sink_detect
from sink_detect import (
    CaptureStore,
    find_image_token_range,
    patch_model,
    rmsnorm,
)


def otsu_threshold(values: np.ndarray) -> float:
    """
    Otsu's method to find optimal binary threshold.
    Maximizes inter-class variance.
    """
    hist, bin_edges = np.histogram(values, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total = hist.sum()
    if total == 0:
        return values.mean()

    sum_all = (hist * bin_centers).sum()
    sum_bg = 0.0
    w_bg = 0.0
    max_var = 0.0
    best_t = bin_centers[0]

    for i in range(len(hist)):
        w_bg += hist[i]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += hist[i] * bin_centers[i]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_all - sum_bg) / w_fg
        var = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if var > max_var:
            max_var = var
            best_t = bin_centers[i]

    return float(best_t)


def plot_phi_distributions(
    hidden_states: List[np.ndarray],
    sink_dims: List[int],
    img_start: int,
    img_end: int,
    output_dir: str,
):
    """
    Per-layer phi distribution with Otsu threshold and
    fixed tau=20 for comparison.
    """
    import matplotlib.pyplot as plt

    n_layers = len(hidden_states)
    sample_layers = [2, 4, 6, 8, 10, 14, 20, 25]
    sample_layers = [l for l in sample_layers if l < n_layers]

    fig, axes = plt.subplots(
        len(sample_layers), 1,
        figsize=(12, 3 * len(sample_layers)),
    )

    adaptive_taus = {}

    for ax_i, li in enumerate(sample_layers):
        hs = hidden_states[li][0]  # [seq, dim]
        rms = np.abs(rmsnorm(hs))
        phi = np.max(
            np.stack([rms[:, d] for d in sink_dims], axis=-1),
            axis=-1,
        )
        vis_phi = phi[img_start:img_end]
        bos_phi = phi[0]

        # Otsu threshold
        tau_otsu = otsu_threshold(vis_phi)
        adaptive_taus[li] = tau_otsu

        n_sink_fixed = (vis_phi > 20).sum()
        n_sink_otsu = (vis_phi > tau_otsu).sum()
        n_total = len(vis_phi)

        ax = axes[ax_i]
        ax.hist(
            vis_phi, bins=50, color="steelblue",
            alpha=0.7, edgecolor="white",
            label=f"visual tokens (n={n_total})",
        )
        ax.axvline(
            x=20, color="red", linestyle="--",
            linewidth=2,
            label=(
                f"τ=20 (paper): "
                f"{n_sink_fixed}/{n_total} sink "
                f"({100*n_sink_fixed/n_total:.0f}%)"
            ),
        )
        ax.axvline(
            x=tau_otsu, color="green", linestyle="-",
            linewidth=2,
            label=(
                f"τ={tau_otsu:.1f} (Otsu): "
                f"{n_sink_otsu}/{n_total} sink "
                f"({100*n_sink_otsu/n_total:.0f}%)"
            ),
        )
        ax.axvline(
            x=bos_phi, color="orange", linestyle=":",
            linewidth=2,
            label=f"BOS φ={bos_phi:.1f}",
        )
        ax.set_ylabel("Count")
        ax.set_title(f"Layer {li}")
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("φ (sink dimension activation)")
    fig.suptitle(
        "Per-Layer φ Distribution: Fixed τ vs Otsu\n"
        f"sink dims = {sink_dims}",
        fontsize=14,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "phi_distributions.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")

    return adaptive_taus


def plot_tau_landscape(
    hidden_states: List[np.ndarray],
    sink_dims: List[int],
    img_start: int,
    img_end: int,
    output_dir: str,
):
    """
    Heatmap: sink percentage for each (layer, tau) combo.
    Shows where the bimodal separation exists.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    n_layers = len(hidden_states)
    tau_range = np.arange(10, 55, 1)
    layers = list(range(2, n_layers))

    sink_pct = np.zeros((len(layers), len(tau_range)))

    for li_idx, li in enumerate(layers):
        hs = hidden_states[li][0]
        rms = np.abs(rmsnorm(hs))
        phi = np.max(
            np.stack([rms[:, d] for d in sink_dims], axis=-1),
            axis=-1,
        )
        vis_phi = phi[img_start:img_end]
        n_total = len(vis_phi)

        for ti, tau in enumerate(tau_range):
            sink_pct[li_idx, ti] = (
                (vis_phi > tau).sum() / n_total * 100
            )

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(
        sink_pct, aspect="auto",
        extent=[
            tau_range[0], tau_range[-1],
            layers[-1], layers[0],
        ],
        cmap="RdYlBu_r",
        vmin=0, vmax=100,
    )
    fig.colorbar(im, ax=ax, label="% Sink Tokens")

    # Mark paper's tau=20
    ax.axvline(x=20, color="white", linestyle="--",
               linewidth=2, label="τ=20 (paper)")

    # Mark the "sweet spot" contour (5% sink)
    cs = ax.contour(
        tau_range, layers,
        sink_pct,
        levels=[5, 10, 50, 90],
        colors=["lime", "yellow", "orange", "red"],
        linewidths=1.5,
    )
    ax.clabel(cs, inline=True, fontsize=9, fmt="%.0f%%")

    ax.set_xlabel("Threshold τ")
    ax.set_ylabel("Layer")
    ax.set_title(
        "Sink Token Percentage by (Layer, τ)\n"
        "Contours at 5%, 10%, 50%, 90%"
    )
    ax.legend(loc="upper right")
    plt.tight_layout()
    path = os.path.join(output_dir, "tau_landscape.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")

    # Print the tau needed for ~5% at each layer
    print("\nτ needed for ~5% sink rate per layer:")
    for li_idx, li in enumerate(layers):
        row = sink_pct[li_idx]
        # Find tau where sink_pct crosses 5%
        below_5 = np.where(row <= 5)[0]
        if len(below_5) > 0:
            tau_5pct = tau_range[below_5[0]]
            print(f"  Layer {li:2d}: τ ≈ {tau_5pct}")
        else:
            print(f"  Layer {li:2d}: τ > {tau_range[-1]} "
                  f"(min sink = {row.min():.1f}%)")


def plot_adaptive_sink_spatial(
    hidden_states: List[np.ndarray],
    sink_dims: List[int],
    img_start: int,
    img_end: int,
    grid_thw: Tuple[int, int, int],
    spatial_merge_size: int,
    video_path: str,
    adaptive_taus: dict,
    output_dir: str,
):
    """
    Show sink spatial maps using adaptive (Otsu) τ
    vs fixed τ=20 side by side.
    """
    import matplotlib.pyplot as plt

    T, H, W = grid_thw
    llm_H = H // spatial_merge_size
    llm_W = W // spatial_merge_size
    n_per_frame = llm_H * llm_W
    n_vis = img_end - img_start

    # Use layer 2 (where τ=20 works) and layer 14
    test_layers = [2, 14]
    test_layers = [
        l for l in test_layers
        if l < len(hidden_states) and l in adaptive_taus
    ]

    # Get frame 0
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h_f, w_f = frame_rgb.shape[:2]

    fig, axes = plt.subplots(
        len(test_layers), 3,
        figsize=(15, 5 * len(test_layers)),
    )
    if len(test_layers) == 1:
        axes = axes.reshape(1, -1)

    for row, li in enumerate(test_layers):
        hs = hidden_states[li][0]
        rms = np.abs(rmsnorm(hs))
        phi = np.max(
            np.stack([rms[:, d] for d in sink_dims], axis=-1),
            axis=-1,
        )
        vis_phi = phi[img_start:img_end]
        frame0_phi = vis_phi[:n_per_frame]
        if len(frame0_phi) != n_per_frame:
            continue
        phi_map = frame0_phi.reshape(llm_H, llm_W)

        tau_otsu = adaptive_taus[li]

        # Col 0: phi heatmap
        phi_up = cv2.resize(
            phi_map.astype(np.float32), (w_f, h_f),
            interpolation=cv2.INTER_NEAREST,
        )
        phi_norm = np.clip(phi_up / 60.0, 0, 1)
        hm = cv2.applyColorMap(
            (phi_norm * 255).astype(np.uint8),
            cv2.COLORMAP_HOT,
        )
        hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
        overlay = (
            0.5 * frame_rgb.astype(np.float32)
            + 0.5 * hm.astype(np.float32)
        ).astype(np.uint8)
        axes[row, 0].imshow(overlay)
        axes[row, 0].set_title(
            f"L{li}: φ heatmap", fontsize=11,
        )
        axes[row, 0].axis("off")

        # Col 1: sink mask at τ=20
        mask_20 = (phi_map > 20).astype(np.float32)
        mask_up = cv2.resize(
            mask_20, (w_f, h_f),
            interpolation=cv2.INTER_NEAREST,
        )
        red_overlay = frame_rgb.copy().astype(np.float32)
        red_overlay[mask_up > 0.5] = (
            red_overlay[mask_up > 0.5] * 0.4
            + np.array([255, 0, 0]) * 0.6
        )
        n20 = int(mask_20.sum())
        axes[row, 1].imshow(red_overlay.astype(np.uint8))
        axes[row, 1].set_title(
            f"L{li}: τ=20 → {n20}/{n_per_frame} sink "
            f"({100*n20/n_per_frame:.0f}%)",
            fontsize=11,
        )
        axes[row, 1].axis("off")

        # Col 2: sink mask at Otsu τ
        mask_otsu = (phi_map > tau_otsu).astype(np.float32)
        mask_up2 = cv2.resize(
            mask_otsu, (w_f, h_f),
            interpolation=cv2.INTER_NEAREST,
        )
        green_overlay = frame_rgb.copy().astype(np.float32)
        green_overlay[mask_up2 > 0.5] = (
            green_overlay[mask_up2 > 0.5] * 0.4
            + np.array([0, 200, 0]) * 0.6
        )
        n_otsu = int(mask_otsu.sum())
        axes[row, 2].imshow(green_overlay.astype(np.uint8))
        axes[row, 2].set_title(
            f"L{li}: τ={tau_otsu:.1f} (Otsu) → "
            f"{n_otsu}/{n_per_frame} sink "
            f"({100*n_otsu/n_per_frame:.0f}%)",
            fontsize=11,
        )
        axes[row, 2].axis("off")

    fig.suptitle(
        "Fixed τ=20 vs Adaptive Otsu τ (Frame 0)\n"
        "Red/Green = sink regions",
        fontsize=14,
    )
    plt.tight_layout()
    path = os.path.join(
        output_dir, "adaptive_comparison.png",
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive sink analysis for Qwen2.5-VL"
    )
    parser.add_argument(
        "--model",
        default=(
            "mlx-community/Qwen2.5-VL-7B-Instruct-8bit"
        ),
    )
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--prompt",
        default="Describe what happens in this video.",
    )
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument(
        "--output-dir", default="sink_adaptive_output",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sink_dims = [458, 2570]

    # ── Load model ──
    from mlx_vlm import load, generate
    from mlx_vlm.utils import load_config
    from mlx_vlm.video_generate import process_vision_info

    print(f"Loading model: {args.model}")
    model, processor = load(args.model)
    config = load_config(args.model)
    patch_model(model)

    # ── Prepare input ──
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
    image_inputs, video_inputs, fps = (
        process_vision_info(messages, True)
    )
    inputs = processor(
        text=[text], images=image_inputs,
        videos=video_inputs, padding=True,
        return_tensors="pt",
    )

    input_ids = mx.array(inputs["input_ids"])
    pixel_values = inputs.get(
        "pixel_values_videos",
        inputs.get("pixel_values", None),
    )
    pixel_values = mx.array(pixel_values)
    mask_input = mx.array(inputs["attention_mask"])

    kwargs = {}
    if inputs.get("video_grid_thw", None) is not None:
        kwargs["video_grid_thw"] = mx.array(
            inputs["video_grid_thw"]
        )
        grid_thw_np = np.array(inputs["video_grid_thw"][0])
    else:
        kwargs["image_grid_thw"] = mx.array(
            inputs["image_grid_thw"]
        )
        grid_thw_np = np.array(inputs["image_grid_thw"][0])

    T, H, W = (
        int(grid_thw_np[0]),
        int(grid_thw_np[1]),
        int(grid_thw_np[2]),
    )
    model_config = model.config
    img_start, img_end = find_image_token_range(
        input_ids, model_config,
    )
    print(f"Grid: T={T}, H={H}, W={W}, "
          f"tokens: [{img_start},{img_end})")

    # ── Run ──
    CaptureStore.enable()
    kwargs["video"] = [args.video]
    kwargs["input_ids"] = input_ids
    kwargs["pixel_values"] = pixel_values
    kwargs["mask"] = mask_input
    kwargs["temperature"] = 0.0
    kwargs["max_tokens"] = args.max_tokens

    output = generate(
        model, processor, prompt=text,
        verbose=False, **kwargs,
    )
    print(f"Output: {output.text[:80]}...")
    CaptureStore.disable()

    hs_np = [np.array(h) for h in CaptureStore.hidden_states]
    print(f"Captured {len(hs_np)} layers")

    # ── Analysis ──
    print("\n" + "=" * 50)
    print("Per-Layer φ Distributions")
    print("=" * 50)
    adaptive_taus = plot_phi_distributions(
        hs_np, sink_dims, img_start, img_end,
        args.output_dir,
    )

    print("\n" + "=" * 50)
    print("τ Landscape Heatmap")
    print("=" * 50)
    plot_tau_landscape(
        hs_np, sink_dims, img_start, img_end,
        args.output_dir,
    )

    print("\n" + "=" * 50)
    print("Adaptive vs Fixed τ Spatial Comparison")
    print("=" * 50)
    spatial_merge = (
        model_config.vision_config.spatial_merge_size
    )
    plot_adaptive_sink_spatial(
        hs_np, sink_dims, img_start, img_end,
        (T, H, W), spatial_merge,
        args.video, adaptive_taus,
        args.output_dir,
    )

    print(f"\nAll outputs → {args.output_dir}/")


if __name__ == "__main__":
    main()
