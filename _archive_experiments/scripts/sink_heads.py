"""
Image-centric head detection + sink token visualization
for Qwen2.5-VL on video, reproducing VisAttnSink paper figs.

Uses fixed τ=20 from the paper to see how the method
behaves on Qwen2.5-VL without modification.

Produces:
  1. Fig3-style: scatter(attn_weight, φ) per layer
  2. Fig4-style: heads sorted by non-sink ratio per layer
  3. Fig1-style: spatial attention maps from last text token
  4. Summary: image-centric head inventory across all layers

Usage:
  uv run --with "transformers<4.52" --with torch \
    --with seaborn --with matplotlib \
    python sink_heads.py \
      --video test_video.mp4 \
      --output-dir sink_heads_output
"""

import argparse
import os
from typing import Dict, List, Tuple

import cv2
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sink_detect import (
    CaptureStore,
    find_image_token_range,
    patch_model,
    rmsnorm,
)


# ── Fixed τ from paper ────────────────────────────────────────

FIXED_TAU = 20.0


def get_phi_and_tau(
    hs: np.ndarray,
    sink_dims: List[int],
    img_start: int,
    img_end: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute φ for all tokens at one layer.
    Returns (full_phi, vis_phi, tau=20).
    """
    rms = np.abs(rmsnorm(hs))
    phi = np.max(
        np.stack([rms[:, d] for d in sink_dims], axis=-1),
        axis=-1,
    )
    vis_phi = phi[img_start:img_end]
    return phi, vis_phi, FIXED_TAU


# ── Fig 3: Scatter (attention weight vs φ) ────────────────────

def fig3_scatter(
    hs_np, aw_np, sink_dims, img_start, img_end,
    output_dir, sample_layers=None,
):
    """
    For each sampled layer, scatter plot of visual tokens:
      x = φ (sink dimension value)
      y = mean attention weight from text tokens
    Color = sink (red) vs non-sink (blue).
    """
    import matplotlib.pyplot as plt

    n_layers = min(len(hs_np), len(aw_np))
    if sample_layers is None:
        sample_layers = [2, 5, 8, 10, 14, 18, 22, 26]
        sample_layers = [l for l in sample_layers
                         if l < n_layers]

    n_cols = 4
    n_rows = (len(sample_layers) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 4 * n_rows),
    )
    axes = axes.flatten()

    for ax_i, li in enumerate(sample_layers):
        phi, vis_phi, tau = get_phi_and_tau(
            hs_np[li][0], sink_dims, img_start, img_end,
        )

        # Attention from text → each visual token
        aw = aw_np[li][0]  # [n_heads, Q, K]
        aw_mean = aw.mean(axis=0)  # [Q, K]
        seq_len = aw_mean.shape[0]
        txt_mask = np.ones(seq_len, dtype=bool)
        txt_mask[img_start:img_end] = False
        # Mean attn from all text queries to each vis token
        txt_rows = aw_mean[txt_mask]  # [n_txt, K]
        vis_attn = txt_rows[:, img_start:img_end].mean(
            axis=0,
        )  # [n_vis]

        is_sink = vis_phi > tau
        n_sink = is_sink.sum()
        n_total = len(vis_phi)

        ax = axes[ax_i]
        ax.scatter(
            vis_phi[~is_sink], vis_attn[~is_sink],
            c="steelblue", s=15, alpha=0.6,
            label=f"non-sink ({n_total - n_sink})",
        )
        ax.scatter(
            vis_phi[is_sink], vis_attn[is_sink],
            c="red", s=15, alpha=0.6,
            label=f"sink ({n_sink})",
        )
        ax.axvline(
            x=tau, color="green", linestyle="--",
            alpha=0.6, linewidth=1.5,
        )
        ax.set_xlabel("φ")
        ax.set_ylabel("Mean txt→vis attn")
        ax.set_title(
            f"Layer {li} (τ={tau:.1f}, "
            f"sink={n_sink}/{n_total})",
            fontsize=10,
        )
        ax.legend(fontsize=7, loc="upper left")

    for i in range(len(sample_layers), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        "Fig 3 style: Attention Weight vs φ "
        "(fixed τ=20 from paper)",
        fontsize=13,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "fig3_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


# ── Fig 4: Heads sorted by non-sink ratio ─────────────────────

def fig4_head_nonsink_ratio(
    hs_np, aw_np, sink_dims, img_start, img_end,
    output_dir, sample_layers=None,
):
    """
    For each layer, compute non-sink ratio per head.
    r = attn_to_nonsink_vis / attn_to_all_vis
    Show as scatter + highlight image-centric heads.
    Also produces a global (layer × head) heatmap.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    n_layers = min(len(hs_np), len(aw_np))
    if sample_layers is None:
        sample_layers = list(range(2, n_layers))

    n_heads = aw_np[2].shape[1]

    # Compute non-sink ratio for all (layer, head) combos
    ratio_map = np.zeros((n_layers, n_heads))
    vis_attn_sum = np.zeros((n_layers, n_heads))
    image_centric = np.zeros((n_layers, n_heads), dtype=bool)

    rho = 0.8  # paper default for general tasks
    summ_thresh = 0.2

    head_info = []  # list of (layer, head, ratio, vis_sum)

    for li in sample_layers:
        phi, vis_phi, tau = get_phi_and_tau(
            hs_np[li][0], sink_dims, img_start, img_end,
        )
        is_sink = vis_phi > tau
        nonsink_local = np.where(~is_sink)[0]
        sink_local = np.where(is_sink)[0]

        aw = aw_np[li][0]  # [n_heads, Q, K]
        seq_len = aw.shape[-1]

        for hi in range(n_heads):
            head_attn = aw[hi]  # [Q, K]

            # Text queries only
            txt_mask = np.ones(seq_len, dtype=bool)
            txt_mask[img_start:img_end] = False
            txt_rows = head_attn[txt_mask]  # [n_txt, K]

            # Sum of attention to ALL visual tokens
            vis_sum = txt_rows[
                :, img_start:img_end
            ].sum(axis=1).mean()
            vis_attn_sum[li, hi] = vis_sum

            if vis_sum < summ_thresh or len(nonsink_local) == 0:
                ratio_map[li, hi] = 0.0
                continue

            # Non-sink ratio
            nonsink_abs = nonsink_local + img_start
            all_vis_attn = txt_rows[
                :, img_start:img_end
            ].sum(axis=1)
            nonsink_attn = txt_rows[
                :, nonsink_abs
            ].sum(axis=1)
            r = (nonsink_attn / (all_vis_attn + 1e-10)).mean()
            ratio_map[li, hi] = r

            if r >= rho and vis_sum >= summ_thresh:
                image_centric[li, hi] = True
                head_info.append((li, hi, float(r), float(vis_sum)))

    # ── Plot 1: Global heatmap ──
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(20, 8),
    )

    im1 = ax1.imshow(
        ratio_map[2:], aspect="auto",
        cmap="RdYlBu", vmin=0, vmax=1,
        extent=[0, n_heads, n_layers - 1, 2],
    )
    fig.colorbar(im1, ax=ax1, label="Non-sink ratio r")
    ax1.set_xlabel("Head")
    ax1.set_ylabel("Layer")
    ax1.set_title(
        f"Non-sink Ratio r (adaptive τ)\n"
        f"ρ={rho}: bright blue = image-centric",
    )

    im2 = ax2.imshow(
        vis_attn_sum[2:], aspect="auto",
        cmap="hot", vmin=0,
        extent=[0, n_heads, n_layers - 1, 2],
    )
    fig.colorbar(im2, ax=ax2, label="Σ vis attn")
    ax2.set_xlabel("Head")
    ax2.set_ylabel("Layer")
    ax2.set_title(
        f"Visual Attention Sum per Head\n"
        f"filter: Σ ≥ {summ_thresh}",
    )

    plt.tight_layout()
    path = os.path.join(output_dir, "fig4_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")

    # ── Plot 2: Scatter like paper's Fig 4 ──
    # For a few representative layers
    scatter_layers = [2, 5, 10, 14, 20, 25]
    scatter_layers = [l for l in scatter_layers
                      if l < n_layers]

    fig, axes = plt.subplots(
        2, 3, figsize=(15, 8),
    )
    axes = axes.flatten()

    for ax_i, li in enumerate(scatter_layers):
        if ax_i >= len(axes):
            break
        ax = axes[ax_i]

        ratios = ratio_map[li]
        vis_sums = vis_attn_sum[li]
        ic = image_centric[li]

        # Color: red=image-centric, grey=filtered, blue=other
        colors = []
        for hi in range(n_heads):
            if ic[hi]:
                colors.append("red")
            elif vis_sums[hi] < summ_thresh:
                colors.append("lightgrey")
            else:
                colors.append("steelblue")

        ax.scatter(
            range(n_heads), ratios,
            c=colors, s=25, alpha=0.7,
        )
        ax.axhline(
            y=rho, color="red", linestyle="--",
            alpha=0.4, linewidth=1,
        )
        n_ic = ic.sum()
        ax.set_title(
            f"Layer {li}: "
            f"{n_ic} image-centric heads",
            fontsize=10,
        )
        ax.set_xlabel("Head index")
        ax.set_ylabel("Non-sink ratio r")
        ax.set_ylim(-0.05, 1.05)

    for i in range(len(scatter_layers), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        "Fig 4 style: Head Non-sink Ratio\n"
        "Red = image-centric (r≥ρ, Σvis≥0.2), "
        "Grey = low visual attention",
        fontsize=13,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "fig4_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")

    # ── Print top image-centric heads ──
    head_info.sort(key=lambda x: -x[2])
    print(f"\nImage-centric heads (r≥{rho}, Σ≥{summ_thresh}):")
    print(f"  Total: {len(head_info)} heads")
    for li, hi, r, vs in head_info[:20]:
        print(f"  Layer {li:2d} Head {hi:2d}: "
              f"r={r:.3f}, Σvis={vs:.3f}")

    return ratio_map, image_centric, head_info


# ── Fig 1/6: Spatial attention maps ───────────────────────────

def fig1_spatial_attn(
    hs_np, aw_np, sink_dims, img_start, img_end,
    grid_thw, spatial_merge_size, video_path,
    image_centric_heads, output_dir,
    sample_layers=None,
):
    """
    For image-centric heads, show spatial attention from
    last text token → visual tokens, overlaid on video frame.
    Marks sink token positions with red boxes.
    """
    import matplotlib.pyplot as plt

    T, H, W = grid_thw
    llm_H = H // spatial_merge_size
    llm_W = W // spatial_merge_size
    n_per_frame = llm_H * llm_W

    # Get frame 0
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Cannot read video frame.")
        return
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h_f, w_f = frame_rgb.shape[:2]

    n_layers = min(len(hs_np), len(aw_np))
    if sample_layers is None:
        sample_layers = [2, 5, 10, 14, 20, 25]
        sample_layers = [l for l in sample_layers
                         if l < n_layers]

    # For each layer, pick top image-centric head
    # Also show a low-ratio head for contrast
    fig, axes = plt.subplots(
        len(sample_layers), 3,
        figsize=(15, 4 * len(sample_layers)),
    )
    if len(sample_layers) == 1:
        axes = axes.reshape(1, -1)

    for row, li in enumerate(sample_layers):
        phi, vis_phi, tau = get_phi_and_tau(
            hs_np[li][0], sink_dims, img_start, img_end,
        )
        is_sink = vis_phi > tau

        aw = aw_np[li][0]  # [n_heads, Q, K]
        seq_len = aw.shape[-1]
        n_heads = aw.shape[0]
        last_txt = seq_len - 1

        # Compute per-head non-sink ratio for this layer
        txt_mask = np.ones(seq_len, dtype=bool)
        txt_mask[img_start:img_end] = False
        nonsink_local = np.where(~is_sink)[0]

        head_ratios = []
        for hi in range(n_heads):
            txt_rows = aw[hi, txt_mask]
            vis_sum = txt_rows[
                :, img_start:img_end
            ].sum(axis=1).mean()
            if vis_sum < 0.2 or len(nonsink_local) == 0:
                head_ratios.append((hi, 0.0, vis_sum))
                continue
            nonsink_abs = nonsink_local + img_start
            all_v = txt_rows[:, img_start:img_end].sum(
                axis=1,
            )
            ns_v = txt_rows[:, nonsink_abs].sum(axis=1)
            r = (ns_v / (all_v + 1e-10)).mean()
            head_ratios.append((hi, float(r), float(vis_sum)))

        head_ratios.sort(key=lambda x: -x[1])
        best_h = head_ratios[0][0]
        best_r = head_ratios[0][1]

        # Find a low-ratio head with decent visual attention
        low_candidates = [
            (h, r, v) for h, r, v in head_ratios
            if v >= 0.1 and r < 0.5
        ]
        if low_candidates:
            low_candidates.sort(key=lambda x: x[1])
            worst_h = low_candidates[0][0]
            worst_r = low_candidates[0][1]
        else:
            worst_h = head_ratios[-1][0]
            worst_r = head_ratios[-1][1]

        # Col 0: φ heatmap with sink mask on frame 0
        frame0_phi = vis_phi[:n_per_frame]
        if len(frame0_phi) == n_per_frame:
            phi_map = frame0_phi.reshape(llm_H, llm_W)
            sink_mask = (phi_map > tau).astype(np.float32)

            phi_up = cv2.resize(
                phi_map.astype(np.float32), (w_f, h_f),
                interpolation=cv2.INTER_NEAREST,
            )
            phi_norm = np.clip(phi_up / (tau * 2), 0, 1)
            hm = cv2.applyColorMap(
                (phi_norm * 255).astype(np.uint8),
                cv2.COLORMAP_HOT,
            )
            hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
            overlay = (
                0.5 * frame_rgb.astype(np.float32)
                + 0.5 * hm.astype(np.float32)
            ).astype(np.uint8)

            # Draw red rectangles on sink cells
            cell_h = h_f / llm_H
            cell_w = w_f / llm_W
            for r_i in range(llm_H):
                for c_i in range(llm_W):
                    if sink_mask[r_i, c_i] > 0.5:
                        y1 = int(r_i * cell_h)
                        y2 = int((r_i + 1) * cell_h)
                        x1 = int(c_i * cell_w)
                        x2 = int((c_i + 1) * cell_w)
                        cv2.rectangle(
                            overlay, (x1, y1), (x2, y2),
                            (255, 0, 0), 2,
                        )

            n_s = int(sink_mask.sum())
            axes[row, 0].imshow(overlay)
            axes[row, 0].set_title(
                f"L{li}: φ map + sink boxes "
                f"(τ={tau:.1f}, {n_s}/{n_per_frame})",
                fontsize=9,
            )
        axes[row, 0].axis("off")

        # Col 1: Best head (high non-sink ratio)
        for col, (head_idx, head_r, label_prefix) in enumerate([
            (best_h, best_r, "Best"),
            (worst_h, worst_r, "Low-r"),
        ], start=1):
            head_attn = aw[head_idx, last_txt,
                           img_start:img_end]
            frame0_attn = head_attn[:n_per_frame]
            if len(frame0_attn) == n_per_frame:
                attn_map = frame0_attn.reshape(llm_H, llm_W)
                attn_up = cv2.resize(
                    attn_map.astype(np.float32), (w_f, h_f),
                    interpolation=cv2.INTER_CUBIC,
                )
                attn_norm = (
                    (attn_up - attn_up.min())
                    / (attn_up.max() - attn_up.min() + 1e-8)
                )
                hm = cv2.applyColorMap(
                    (attn_norm * 255).astype(np.uint8),
                    cv2.COLORMAP_JET,
                )
                hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
                overlay = (
                    0.5 * frame_rgb.astype(np.float32)
                    + 0.5 * hm.astype(np.float32)
                ).astype(np.uint8)
                axes[row, col].imshow(overlay)
                axes[row, col].set_title(
                    f"L{li} H{head_idx} "
                    f"({label_prefix}, r={head_r:.2f})",
                    fontsize=9,
                )
            axes[row, col].axis("off")

        axes[row, 0].set_ylabel(
            f"L{li}", fontsize=12, rotation=0,
            labelpad=30, va="center",
        )

    fig.suptitle(
        "Fig 1/6 style: Sink Positions + "
        "Head Attention Maps\n"
        "Col 0 = φ + sink boxes, "
        "Col 1 = best head, Col 2 = low-r head",
        fontsize=13,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "fig1_spatial.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


# ── Main ──────────────────────────────────────────────────────

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
        default="Describe what happens in this video.",
    )
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument(
        "--output-dir", default="sink_heads_output",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sink_dims = [458, 2570]

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

    kwargs = {}
    for key in ["video_grid_thw", "image_grid_thw"]:
        if inputs.get(key) is not None:
            kwargs[key] = mx.array(inputs[key])
            grid_thw_np = np.array(inputs[key][0])

    T, H, W = (
        int(grid_thw_np[0]),
        int(grid_thw_np[1]),
        int(grid_thw_np[2]),
    )
    model_config = model.config
    img_start, img_end = find_image_token_range(
        input_ids, model_config,
    )
    spatial_merge = (
        model_config.vision_config.spatial_merge_size
    )
    print(f"Grid: T={T}, H={H}, W={W}, "
          f"vis: [{img_start},{img_end})")

    CaptureStore.enable()
    gen_kwargs = dict(kwargs)
    gen_kwargs["video"] = [args.video]
    gen_kwargs["input_ids"] = input_ids
    gen_kwargs["pixel_values"] = pixel_values
    gen_kwargs["mask"] = mask_input
    gen_kwargs["temperature"] = 0.0
    gen_kwargs["max_tokens"] = args.max_tokens

    output = generate(
        model, processor, prompt=text,
        verbose=False, **gen_kwargs,
    )
    print(f"Output: {output.text[:80]}...")
    CaptureStore.disable()

    hs_np = [np.array(h) for h in CaptureStore.hidden_states]
    aw_np = [np.array(a) for a in CaptureStore.attn_weights]
    print(f"Captured {len(hs_np)} layers "
          f"(hs + attn weights)")

    # ── Visualizations ──
    print("\n" + "=" * 50)
    print("Fig 3: Scatter (attn weight vs φ)")
    print("=" * 50)
    fig3_scatter(
        hs_np, aw_np, sink_dims,
        img_start, img_end, args.output_dir,
    )

    print("\n" + "=" * 50)
    print("Fig 4: Head non-sink ratio")
    print("=" * 50)
    ratio_map, ic_map, head_info = fig4_head_nonsink_ratio(
        hs_np, aw_np, sink_dims,
        img_start, img_end, args.output_dir,
    )

    print("\n" + "=" * 50)
    print("Fig 1/6: Spatial attention maps")
    print("=" * 50)
    fig1_spatial_attn(
        hs_np, aw_np, sink_dims,
        img_start, img_end,
        (T, H, W), spatial_merge,
        args.video, ic_map, args.output_dir,
    )

    print(f"\nAll outputs → {args.output_dir}/")


if __name__ == "__main__":
    main()
