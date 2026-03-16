"""
Cross-frame spatial analysis of visual attention sinks in video.

Key questions (video-specific, not applicable to single images):
  1. Do sink tokens occupy the same spatial positions across frames?
  2. How does cross-frame consistency vary by layer?
  3. Are sink positions content-dependent or position-dependent?
  4. How does the sink spatial pattern evolve across layers?

Usage:
  uv run --with "transformers<4.52" --with torch \
    --with seaborn --with matplotlib \
    python sink_spatial_video.py \
      --video test_video.mp4 \
      --output-dir sink_spatial_output
"""

import argparse
import os
from typing import Dict, List, Tuple

import cv2
import mlx.core as mx
import numpy as np

from sink_detect import (
    CaptureStore,
    find_image_token_range,
    patch_model,
    rmsnorm,
)


def get_per_frame_sink_maps(
    hidden_states: List[np.ndarray],
    sink_dims: List[int],
    img_start: int,
    img_end: int,
    llm_H: int,
    llm_W: int,
    tau: float = 20.0,
) -> Dict[int, List[np.ndarray]]:
    """
    For each layer, compute per-frame binary sink maps.
    Returns {layer_idx: [frame0_map, frame1_map, ...]}.
    Each map is shape (llm_H, llm_W), binary.
    """
    n_per_frame = llm_H * llm_W
    n_vis = img_end - img_start
    n_frames = n_vis // n_per_frame
    n_layers = len(hidden_states)

    result = {}
    for li in range(2, n_layers):
        hs = hidden_states[li][0]
        rms = np.abs(rmsnorm(hs))
        phi = np.max(
            np.stack(
                [rms[:, d] for d in sink_dims], axis=-1
            ),
            axis=-1,
        )
        vis_phi = phi[img_start:img_end]

        frame_maps = []
        for fi in range(n_frames):
            fp = vis_phi[
                fi * n_per_frame:(fi + 1) * n_per_frame
            ]
            if len(fp) == n_per_frame:
                frame_maps.append(
                    (fp > tau).astype(np.float32).reshape(
                        llm_H, llm_W
                    )
                )
        result[li] = frame_maps
    return result


def get_per_frame_phi_maps(
    hidden_states: List[np.ndarray],
    sink_dims: List[int],
    img_start: int,
    img_end: int,
    llm_H: int,
    llm_W: int,
) -> Dict[int, List[np.ndarray]]:
    """
    For each layer, compute per-frame continuous phi maps.
    Returns {layer_idx: [frame0_phi, frame1_phi, ...]}.
    """
    n_per_frame = llm_H * llm_W
    n_vis = img_end - img_start
    n_frames = n_vis // n_per_frame
    n_layers = len(hidden_states)

    result = {}
    for li in range(2, n_layers):
        hs = hidden_states[li][0]
        rms = np.abs(rmsnorm(hs))
        phi = np.max(
            np.stack(
                [rms[:, d] for d in sink_dims], axis=-1
            ),
            axis=-1,
        )
        vis_phi = phi[img_start:img_end]

        frame_maps = []
        for fi in range(n_frames):
            fp = vis_phi[
                fi * n_per_frame:(fi + 1) * n_per_frame
            ]
            if len(fp) == n_per_frame:
                frame_maps.append(
                    fp.reshape(llm_H, llm_W)
                )
        result[li] = frame_maps
    return result


def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = (a * b).sum()
    union = np.clip(a + b, 0, 1).sum()
    return float(inter / union) if union > 0 else 0.0


# ── Plot 1: IoU across all layers ────────────────────────────

def plot_iou_by_layer(
    sink_maps: Dict[int, List[np.ndarray]],
    output_dir: str,
):
    """
    Mean pairwise IoU of sink positions across frames,
    plotted for every layer. Shows at which layers sink
    positions are most consistent across frames.
    """
    import matplotlib.pyplot as plt

    layers = sorted(sink_maps.keys())
    mean_ious = []
    min_ious = []
    max_ious = []

    for li in layers:
        maps = sink_maps[li]
        n = len(maps)
        if n < 2:
            mean_ious.append(0)
            min_ious.append(0)
            max_ious.append(0)
            continue
        ious = []
        for i in range(n):
            for j in range(i + 1, n):
                ious.append(compute_iou(maps[i], maps[j]))
        mean_ious.append(np.mean(ious))
        min_ious.append(np.min(ious))
        max_ious.append(np.max(ious))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(
        layers, min_ious, max_ious,
        alpha=0.2, color="coral",
        label="min–max range",
    )
    ax.plot(
        layers, mean_ious, "r-o",
        markersize=4, linewidth=2,
        label="mean IoU",
    )
    ax.axhline(y=0.5, color="grey", linestyle="--",
               alpha=0.5)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Cross-frame Sink IoU", fontsize=12)
    ax.set_title(
        "Sink Spatial Consistency Across Video Frames\n"
        "(IoU of binary sink maps between frame pairs)",
        fontsize=13,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "iou_by_layer.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


# ── Plot 2: Frequency maps across layers ─────────────────────

def plot_frequency_maps(
    sink_maps: Dict[int, List[np.ndarray]],
    output_dir: str,
):
    """
    For each sampled layer, show the sink frequency map
    (how often each spatial position is a sink across frames).
    """
    import matplotlib.pyplot as plt

    sample_layers = [2, 5, 8, 10, 14, 18, 22, 26]
    sample_layers = [
        l for l in sample_layers if l in sink_maps
    ]

    n = len(sample_layers)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(4 * cols, 3.5 * rows),
    )
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, li in enumerate(sample_layers):
        r, c = divmod(idx, cols)
        maps = sink_maps[li]
        freq = np.mean(maps, axis=0)
        n_frames = len(maps)
        mean_sink_pct = np.mean(
            [m.mean() for m in maps]
        ) * 100

        ax = axes[r, c]
        im = ax.imshow(
            freq, cmap="YlOrRd", vmin=0, vmax=1,
            interpolation="nearest",
        )
        ax.set_title(
            f"Layer {li}\n"
            f"sink={mean_sink_pct:.0f}%, "
            f"{n_frames} frames",
            fontsize=10,
        )
        ax.set_xlabel("W")
        ax.set_ylabel("H")

    # Hide unused axes
    for idx in range(len(sample_layers), rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    fig.colorbar(
        im, ax=axes, shrink=0.6,
        label="Fraction of frames where position is sink",
    )
    fig.suptitle(
        "Sink Frequency Maps Across Layers\n"
        "bright = sink in most/all frames",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(
        output_dir, "freq_maps_by_layer.png",
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


# ── Plot 3: Per-frame sink maps at key layers ────────────────

def plot_perframe_sink_grid(
    sink_maps: Dict[int, List[np.ndarray]],
    phi_maps: Dict[int, List[np.ndarray]],
    frames_rgb: List[np.ndarray],
    output_dir: str,
):
    """
    Grid: rows = layers, cols = frames.
    Each cell shows the sink binary mask overlaid on the
    video frame. Allows visual comparison of which spatial
    regions are sinks across frames and layers.
    """
    import matplotlib.pyplot as plt

    sample_layers = [5, 10, 14, 20, 26]
    sample_layers = [
        l for l in sample_layers if l in sink_maps
    ]

    n_frames = min(len(frames_rgb), 6)
    fig, axes = plt.subplots(
        len(sample_layers), n_frames,
        figsize=(3 * n_frames, 3 * len(sample_layers)),
    )
    if len(sample_layers) == 1:
        axes = axes.reshape(1, -1)
    if n_frames == 1:
        axes = axes.reshape(-1, 1)

    for row, li in enumerate(sample_layers):
        maps = sink_maps[li]
        phis = phi_maps[li]

        for col in range(n_frames):
            ax = axes[row, col]
            if col >= len(maps) or col >= len(frames_rgb):
                ax.axis("off")
                continue

            frame = frames_rgb[col]
            h_f, w_f = frame.shape[:2]
            sink_mask = maps[col]
            phi_map = phis[col]
            llm_H, llm_W = sink_mask.shape

            # Upsample sink mask to frame size
            mask_up = cv2.resize(
                sink_mask,
                (w_f, h_f),
                interpolation=cv2.INTER_NEAREST,
            )

            # Create colored overlay: red for sink,
            # green for non-sink
            overlay = frame.copy().astype(np.float32)
            # Tint sink regions red
            overlay[mask_up > 0.5, 0] = np.clip(
                overlay[mask_up > 0.5, 0] * 0.5 + 128,
                0, 255,
            )
            overlay[mask_up > 0.5, 1] *= 0.4
            overlay[mask_up > 0.5, 2] *= 0.4
            # Tint non-sink regions green
            overlay[mask_up < 0.5, 0] *= 0.5
            overlay[mask_up < 0.5, 1] = np.clip(
                overlay[mask_up < 0.5, 1] * 0.5 + 80,
                0, 255,
            )
            overlay[mask_up < 0.5, 2] *= 0.5

            ax.imshow(overlay.astype(np.uint8))
            n_sink = int(sink_mask.sum())
            n_total = llm_H * llm_W
            ax.set_title(
                f"F{col}: {n_sink}/{n_total} sink",
                fontsize=8,
            )
            ax.axis("off")

        # Add row label on leftmost axis
        axes[row, 0].text(
            -0.15, 0.5, f"Layer {li}",
            transform=axes[row, 0].transAxes,
            fontsize=13, fontweight="bold",
            va="center", ha="right",
            rotation=90,
        )

    fig.suptitle(
        "Sink Spatial Maps: Red = Sink, Green = Non-Sink\n"
        "rows = layers, columns = video frames",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(
        output_dir, "perframe_sink_grid.png",
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


# ── Plot 4: Position-dependent vs content-dependent ──────────

def plot_position_vs_content(
    sink_maps: Dict[int, List[np.ndarray]],
    phi_maps: Dict[int, List[np.ndarray]],
    output_dir: str,
):
    """
    For each spatial position (h, w), compute:
      - Across-frame variance of phi (content sensitivity)
      - Across-frame mean of phi (positional bias)

    If sink is position-dependent: high mean, low variance.
    If sink is content-dependent: high variance.
    """
    import matplotlib.pyplot as plt

    sample_layers = [5, 10, 14, 20]
    sample_layers = [
        l for l in sample_layers if l in phi_maps
    ]

    fig, axes = plt.subplots(
        len(sample_layers), 3,
        figsize=(14, 3.5 * len(sample_layers)),
    )
    if len(sample_layers) == 1:
        axes = axes.reshape(1, -1)

    for row, li in enumerate(sample_layers):
        phis = phi_maps[li]
        # Stack: [n_frames, H, W]
        phi_stack = np.stack(phis, axis=0)

        mean_phi = phi_stack.mean(axis=0)
        std_phi = phi_stack.std(axis=0)
        # Coefficient of variation
        cv_phi = std_phi / np.clip(mean_phi, 1e-6, None)

        # Mean phi map
        ax = axes[row, 0]
        im0 = ax.imshow(
            mean_phi, cmap="hot",
            interpolation="nearest",
        )
        ax.set_title(
            f"Layer {li}: Mean φ across frames",
            fontsize=10,
        )
        fig.colorbar(im0, ax=ax, shrink=0.8)

        # Std phi map
        ax = axes[row, 1]
        im1 = ax.imshow(
            std_phi, cmap="viridis",
            interpolation="nearest",
        )
        ax.set_title(
            f"Layer {li}: Std φ across frames",
            fontsize=10,
        )
        fig.colorbar(im1, ax=ax, shrink=0.8)

        # CV map
        ax = axes[row, 2]
        im2 = ax.imshow(
            cv_phi, cmap="coolwarm", vmin=0, vmax=0.5,
            interpolation="nearest",
        )
        ax.set_title(
            f"Layer {li}: CV(φ) = σ/μ\n"
            f"low=positional, high=content",
            fontsize=10,
        )
        fig.colorbar(im2, ax=ax, shrink=0.8)

    fig.suptitle(
        "Position-Dependent vs Content-Dependent Sinks\n"
        "Low CV → sink is position-dependent "
        "(same spots always sink)\n"
        "High CV → sink is content-dependent "
        "(varies with frame content)",
        fontsize=12,
    )
    plt.tight_layout()
    path = os.path.join(
        output_dir, "position_vs_content.png",
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")

    # Print summary
    for li in sample_layers:
        phis = phi_maps[li]
        phi_stack = np.stack(phis, axis=0)
        mean_phi = phi_stack.mean(axis=0)
        cv_phi = phi_stack.std(axis=0) / np.clip(
            mean_phi, 1e-6, None
        )
        print(
            f"  Layer {li}: "
            f"median CV = {np.median(cv_phi):.3f}, "
            f"mean CV = {np.mean(cv_phi):.3f}"
        )


# ── Plot 5: Temporal drift of sink positions ─────────────────

def plot_temporal_drift(
    sink_maps: Dict[int, List[np.ndarray]],
    output_dir: str,
):
    """
    For a fixed layer, compute IoU between frame 0 and each
    subsequent frame. Shows if sink positions drift over time.
    """
    import matplotlib.pyplot as plt

    test_layers = [5, 10, 14, 20]
    test_layers = [
        l for l in test_layers if l in sink_maps
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    for li in test_layers:
        maps = sink_maps[li]
        n = len(maps)
        if n < 2:
            continue
        ious_vs_f0 = [
            compute_iou(maps[0], maps[fi])
            for fi in range(n)
        ]
        ax.plot(
            range(n), ious_vs_f0, "-o",
            markersize=5, label=f"Layer {li}",
        )

    ax.set_xlabel("Frame index", fontsize=12)
    ax.set_ylabel("IoU vs Frame 0", fontsize=12)
    ax.set_title(
        "Temporal Drift of Sink Positions\n"
        "(IoU between frame 0 and each frame)",
        fontsize=13,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(
        output_dir, "temporal_drift.png",
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


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
        "--max-tokens", type=int, default=50,
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument(
        "--output-dir", default="sink_spatial_output",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sink_dims = [458, 2570]

    # ── Load & run ──
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
    if inputs.get("video_grid_thw") is not None:
        kwargs["video_grid_thw"] = mx.array(
            inputs["video_grid_thw"]
        )
        grid_thw = [
            int(x) for x in inputs["video_grid_thw"][0]
        ]
    elif inputs.get("image_grid_thw") is not None:
        kwargs["image_grid_thw"] = mx.array(
            inputs["image_grid_thw"]
        )
        grid_thw = [
            int(x) for x in inputs["image_grid_thw"][0]
        ]

    T, H, W = grid_thw
    model_config = model.config
    spatial_merge = (
        model_config.vision_config.spatial_merge_size
    )
    llm_H = H // spatial_merge
    llm_W = W // spatial_merge

    img_start, img_end = find_image_token_range(
        input_ids, model_config,
    )
    n_vis = img_end - img_start
    n_per_frame = llm_H * llm_W
    n_frames = n_vis // n_per_frame

    print(f"Grid: T={T}, H={H}, W={W}")
    print(f"LLM spatial: {llm_H}×{llm_W} = "
          f"{n_per_frame} tokens/frame")
    print(f"Vis tokens: {n_vis}, frames: {n_frames}")

    # Run inference
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

    # ── Compute maps ──
    print("\nComputing per-frame sink maps...")
    sink_maps = get_per_frame_sink_maps(
        hs_np, sink_dims, img_start, img_end,
        llm_H, llm_W,
    )
    phi_maps = get_per_frame_phi_maps(
        hs_np, sink_dims, img_start, img_end,
        llm_H, llm_W,
    )

    # Extract video frames for overlay
    cap = cv2.VideoCapture(args.video)
    total_vid_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    step = max(1, total_vid_frames // n_frames)
    frames_rgb = []
    for fi in range(min(n_frames, 8)):
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            min(fi * step, total_vid_frames - 1),
        )
        ret, frame = cap.read()
        if ret:
            frames_rgb.append(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
    cap.release()

    # ── Generate all plots ──
    print("\n" + "=" * 50)
    print("Generating cross-frame spatial analysis...")
    print("=" * 50)

    plot_iou_by_layer(sink_maps, args.output_dir)
    plot_frequency_maps(sink_maps, args.output_dir)
    plot_perframe_sink_grid(
        sink_maps, phi_maps, frames_rgb, args.output_dir,
    )
    plot_position_vs_content(
        sink_maps, phi_maps, args.output_dir,
    )
    plot_temporal_drift(sink_maps, args.output_dir)

    print(f"\nAll outputs → {args.output_dir}/")


if __name__ == "__main__":
    main()
