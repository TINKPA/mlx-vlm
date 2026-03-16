"""
Attention Visualization for Qwen2.5-VL on MLX.

Three-stage pipeline:
  Stage 1: Layer × Modality Flow heatmap
  Stage 2: Head Specialization scatter plot
  Stage 3: Spatio-Temporal attention overlay

Usage:
  uv run --with "transformers<4.52" --with torch \
    --with seaborn --with matplotlib \
    python attn_viz.py \
      --model mlx-community/Qwen2.5-VL-7B-Instruct-8bit \
      --video test_video.mp4 \
      --prompt "Describe what happens in this video." \
      --output-dir attn_output
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image


# ── Attention capture via monkey-patch ──────────────────────────

class AttentionCapture:
    """Global store for captured attention weights."""
    weights: List[mx.array] = []  # one per layer
    enabled: bool = False
    image_token_range: Optional[Tuple[int, int]] = None

    @classmethod
    def reset(cls):
        cls.weights = []

    @classmethod
    def enable(cls):
        cls.enabled = True
        cls.reset()

    @classmethod
    def disable(cls):
        cls.enabled = False


def _patched_attention_call(self, x, mask=None, cache=None,
                            position_ids=None):
    """
    Drop-in replacement for Attention.__call__ in language.py.
    Computes attention manually to capture weights.
    """
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

    if position_ids is None:
        kv_seq_len = keys.shape[-2]
        kv_seq_len += cache.offset + 1
        position_ids = mx.arange(cache.offset, cache.offset + L)
        position_ids = mx.expand_dims(position_ids, axis=0)
        position_ids = mx.tile(position_ids, (3, 1, 1))
    else:
        kv_seq_len = keys.shape[-2]
        kv_seq_len += cache.offset + 1 if cache is not None else 0

    # Import from the module where this Attention class lives
    from mlx_vlm.models.qwen2_5_vl.language import (
        apply_multimodal_rotary_pos_emb,
    )
    cos, sin = self.rotary_emb(values, position_ids)

    if mask is not None and isinstance(mask, mx.array):
        mask = mask[..., :keys.shape[-2]]

    queries, keys = apply_multimodal_rotary_pos_emb(
        queries, keys, cos, sin, unqueeze_dim=1
    )

    if cache is not None:
        keys, values = cache.update_and_fetch(keys, values)

    # ── Manual attention (the key change) ──
    # GQA: expand kv heads to match query heads
    n_rep = self.n_heads // self.n_kv_heads
    if n_rep > 1:
        keys = mx.repeat(keys, n_rep, axis=1)
        values = mx.repeat(values, n_rep, axis=1)

    if AttentionCapture.enabled:
        # Manual path: compute and store attention weights
        # queries: [B, n_heads, L_q, head_dim]
        # keys:    [B, n_heads, L_k, head_dim]
        attn_scores = (
            queries @ keys.transpose(0, 1, 3, 2)
        ) * self.scale  # [B, n_heads, L_q, L_k]

        if mask is not None and isinstance(mask, mx.array):
            attn_scores = attn_scores + mask
        elif mask is not None and mask == "causal":
            # Build causal mask manually
            L_q = attn_scores.shape[2]
            L_k = attn_scores.shape[3]
            causal = mx.triu(
                mx.full((L_q, L_k), float("-inf")), k=1
            )
            attn_scores = attn_scores + causal

        attn_weights = mx.softmax(attn_scores, axis=-1)

        # Store only for the prefill pass (L > 1), not
        # autoregressive decode steps
        if L > 1:
            AttentionCapture.weights.append(attn_weights)

        output = attn_weights @ values
    else:
        # Fast fused path (normal)
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values,
            scale=self.scale, mask=mask
        )

    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(output)


def patch_model(model):
    """Monkey-patch Attention class to capture weights."""
    from mlx_vlm.models.qwen2_5_vl.language import Attention
    Attention.__call__ = _patched_attention_call
    n_layers = len(model.language_model.model.layers)
    print(f"Patched Attention class ({n_layers} layers).")


# ── Token boundary detection ──────────────────────────────────

def find_image_token_range(input_ids, config):
    """Find [start, end) indices of image/video tokens."""
    ids = input_ids.flatten().tolist()
    img_id = config.image_token_id
    vid_id = config.video_token_id

    token_id = img_id if img_id in ids else vid_id
    positions = [i for i, t in enumerate(ids) if t == token_id]
    if not positions:
        raise ValueError("No image/video tokens found in input_ids")
    return positions[0], positions[-1] + 1


# ── Stage 1: Modality Flow Heatmap ────────────────────────────

def stage1_modality_flow(
    attn_weights: List[np.ndarray],
    img_start: int,
    img_end: int,
    output_dir: str,
):
    """
    Compute per-layer modality flow and plot heatmap.
    attn_weights: list of [1, n_heads, Q, K] per layer
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_layers = len(attn_weights)
    flows = np.zeros((n_layers, 4))
    labels = ["txt→txt", "txt→img", "img→txt", "img→img"]

    for li, aw in enumerate(attn_weights):
        # aw: [1, n_heads, Q, K] → average over heads
        a = aw[0].mean(axis=0)  # [Q, K]
        seq_len = a.shape[0]

        # Masks
        txt_q = np.ones(seq_len, dtype=bool)
        txt_q[img_start:img_end] = False
        img_q = ~txt_q
        txt_k = txt_q.copy()
        img_k = img_q.copy()

        # Normalize each query's attention sums to 1
        # then average across queries in each group
        def flow(q_mask, k_mask):
            rows = a[q_mask]
            if rows.shape[0] == 0:
                return 0.0
            return rows[:, k_mask].sum(axis=1).mean()

        flows[li, 0] = flow(txt_q, txt_k)
        flows[li, 1] = flow(txt_q, img_k)
        flows[li, 2] = flow(img_q, txt_k)
        flows[li, 3] = flow(img_q, img_k)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 10))
    sns.heatmap(
        flows,
        annot=True, fmt=".2f",
        xticklabels=labels,
        yticklabels=[f"L{i}" for i in range(n_layers)],
        cmap="YlOrRd",
        vmin=0, vmax=1,
        ax=ax,
    )
    ax.set_title("Stage 1: Layer-wise Modality Attention Flow")
    ax.set_xlabel("Attention Flow Direction")
    ax.set_ylabel("Layer")
    plt.tight_layout()
    path = os.path.join(output_dir, "stage1_modality_flow.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Stage 1 saved → {path}")
    return flows


# ── Stage 2: Head Specialization Scatter ──────────────────────

def stage2_head_scatter(
    attn_weights: List[np.ndarray],
    img_start: int,
    img_end: int,
    output_dir: str,
    threshold: float = 0.4,
):
    """Scatter plot of txt→img attention ratio per head."""
    import matplotlib.pyplot as plt

    n_layers = len(attn_weights)
    n_heads = attn_weights[0].shape[1]
    seq_len = attn_weights[0].shape[-1]

    txt_q = np.ones(seq_len, dtype=bool)
    txt_q[img_start:img_end] = False
    img_k = ~txt_q.copy()

    points_layer = []
    points_head = []
    points_ratio = []

    for li, aw in enumerate(attn_weights):
        a = aw[0]  # [n_heads, Q, K]
        for hi in range(n_heads):
            head_attn = a[hi]  # [Q, K]
            txt_rows = head_attn[txt_q]
            if txt_rows.shape[0] == 0:
                continue
            ratio = txt_rows[:, img_k].sum(axis=1).mean()
            points_layer.append(li)
            points_head.append(hi)
            points_ratio.append(ratio)

    points_layer = np.array(points_layer)
    points_ratio = np.array(points_ratio)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [
        "red" if r > threshold else "steelblue"
        for r in points_ratio
    ]
    ax.scatter(
        points_layer, points_ratio,
        c=colors, alpha=0.5, s=20, edgecolors="none",
    )
    ax.axhline(
        y=threshold, color="red",
        linestyle="--", alpha=0.5, label=f"threshold={threshold}",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("txt→img Attention Ratio")
    ax.set_title("Stage 2: Head Specialization "
                 "(red = visual-tracking heads)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "stage2_head_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Stage 2 saved → {path}")

    # Return top visual heads: (layer, head, ratio)
    top_idx = np.argsort(points_ratio)[::-1][:10]
    top_heads = [
        (points_layer[i], points_head[i], points_ratio[i])
        for i in top_idx
    ]
    print("Top 10 visual heads (layer, head, txt→img ratio):")
    for l, h, r in top_heads:
        print(f"  Layer {l:2d}, Head {h:2d}: {r:.3f}")
    return top_heads


# ── Stage 3: Spatio-Temporal Overlay ──────────────────────────

def stage3_spatiotemporal(
    attn_weights: List[np.ndarray],
    img_start: int,
    img_end: int,
    top_heads: list,
    video_path: str,
    grid_thw: tuple,
    output_dir: str,
    spatial_merge_size: int = 2,
):
    """
    For the best visual head, produce:
      (a) Temporal attention bar chart
      (b) Top-3 frame spatial overlays
    """
    import matplotlib.pyplot as plt

    if not top_heads:
        print("No visual heads found, skipping Stage 3.")
        return

    best_layer, best_head, best_ratio = top_heads[0]
    print(f"\nStage 3 using Layer {best_layer}, "
          f"Head {best_head} (ratio={best_ratio:.3f})")

    # Get attention from last text token → image tokens
    aw = attn_weights[best_layer][0]  # [n_heads, Q, K]
    head_attn = aw[best_head]  # [Q, K]

    # Use the last token before generation as the query
    # (the last text token, which is after image tokens)
    seq_len = head_attn.shape[0]
    last_txt_idx = seq_len - 1
    img_attn = head_attn[last_txt_idx, img_start:img_end]

    # Parse grid_thw
    T, H, W = grid_thw
    llm_H = H // spatial_merge_size
    llm_W = W // spatial_merge_size
    n_tokens_per_frame = llm_H * llm_W
    n_frames = T

    expected_tokens = n_frames * n_tokens_per_frame
    actual_tokens = img_attn.shape[0]
    print(f"  Grid: T={T}, H={H}, W={W} → "
          f"LLM grid {n_frames}×{llm_H}×{llm_W} "
          f"= {expected_tokens} tokens "
          f"(actual: {actual_tokens})")

    if actual_tokens != expected_tokens:
        print(f"  Warning: token count mismatch, "
              f"adjusting n_frames")
        n_frames = actual_tokens // n_tokens_per_frame
        img_attn = img_attn[:n_frames * n_tokens_per_frame]

    # Reshape to [T, H, W]
    spatial_attn = img_attn.reshape(n_frames, llm_H, llm_W)

    # (a) Temporal attention bar chart
    temporal_attn = spatial_attn.sum(axis=(1, 2))
    temporal_attn = temporal_attn / temporal_attn.sum()

    fig, ax = plt.subplots(figsize=(10, 3))
    bars = ax.bar(range(n_frames), temporal_attn, color="steelblue")

    # Highlight top-3
    top3_frames = np.argsort(temporal_attn)[::-1][:3]
    for fi in top3_frames:
        bars[fi].set_color("red")

    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Attention Weight (normalized)")
    ax.set_title(f"Stage 3a: Temporal Attention "
                 f"(L{best_layer} H{best_head})")
    plt.tight_layout()
    path = os.path.join(output_dir, "stage3a_temporal.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Stage 3a saved → {path}")

    # (b) Extract actual frames from video and overlay
    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Map model frame indices to video frame indices
    frame_step = max(1, total_video_frames // n_frames)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax_i, fi in enumerate(top3_frames):
        video_frame_idx = min(fi * frame_step,
                              total_video_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h_frame, w_frame = frame_rgb.shape[:2]

        # Get spatial attention for this frame, upsample
        frame_attn = spatial_attn[fi]  # [llm_H, llm_W]
        frame_attn = (
            frame_attn - frame_attn.min()
        ) / (frame_attn.max() - frame_attn.min() + 1e-8)
        heatmap = cv2.resize(
            frame_attn.astype(np.float32),
            (w_frame, h_frame),
            interpolation=cv2.INTER_CUBIC,
        )
        heatmap_color = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        overlay = (
            0.5 * frame_rgb.astype(np.float32)
            + 0.5 * heatmap_color.astype(np.float32)
        ).astype(np.uint8)

        axes[ax_i].imshow(overlay)
        t_sec = video_frame_idx / fps if fps > 0 else 0
        axes[ax_i].set_title(
            f"Frame {fi} ({t_sec:.1f}s) "
            f"attn={temporal_attn[fi]:.3f}"
        )
        axes[ax_i].axis("off")

    cap.release()
    fig.suptitle(
        f"Stage 3b: Spatial Attention Overlay "
        f"(L{best_layer} H{best_head})",
        fontsize=14,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "stage3b_spatial_overlay.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Stage 3b saved → {path}")


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Attention visualization for Qwen2.5-VL"
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-VL-7B-Instruct-8bit",
    )
    parser.add_argument(
        "--video", type=str, required=True,
    )
    parser.add_argument(
        "--prompt",
        default="Describe what happens in this video.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=50,
    )
    parser.add_argument(
        "--fps", type=float, default=1.0,
    )
    parser.add_argument(
        "--output-dir", default="attn_output",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.4,
        help="Threshold for visual head detection",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ──
    from mlx_vlm import load, generate
    from mlx_vlm.utils import load_config
    from mlx_vlm.video_generate import (
        process_vision_info,
    )

    print(f"Loading model: {args.model}")
    model, processor = load(args.model)
    config = load_config(args.model)

    # ── Patch attention layers ──
    patch_model(model)

    # ── Prepare video input (reuse video_generate logic) ──
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
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs, fps = process_vision_info(
        messages, True,
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    input_ids = mx.array(inputs["input_ids"])
    pixel_values = inputs.get(
        "pixel_values_videos",
        inputs.get("pixel_values", None),
    )
    if pixel_values is None:
        raise ValueError("No pixel values found.")
    pixel_values = mx.array(pixel_values)
    mask_input = mx.array(inputs["attention_mask"])

    # Get grid_thw for spatial mapping
    kwargs = {}
    if inputs.get("video_grid_thw", None) is not None:
        kwargs["video_grid_thw"] = mx.array(
            inputs["video_grid_thw"]
        )
        grid_thw_np = np.array(inputs["video_grid_thw"][0])
    elif inputs.get("image_grid_thw", None) is not None:
        kwargs["image_grid_thw"] = mx.array(
            inputs["image_grid_thw"]
        )
        grid_thw_np = np.array(inputs["image_grid_thw"][0])
    else:
        raise ValueError("No grid_thw found.")

    T, H, W = int(grid_thw_np[0]), int(grid_thw_np[1]), \
        int(grid_thw_np[2])
    print(f"Video grid: T={T}, H={H}, W={W}")

    # Find image token boundaries
    model_config = model.config
    img_start, img_end = find_image_token_range(
        input_ids, model_config,
    )
    print(f"Image tokens: [{img_start}, {img_end}) "
          f"= {img_end - img_start} tokens")
    print(f"Total sequence length: {input_ids.shape[1]}")

    # ── Run prefill with attention capture ──
    print("\nRunning prefill with attention capture...")
    AttentionCapture.enable()

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
    print(f"Model output: {output.text}")

    AttentionCapture.disable()

    # ── Convert to numpy ──
    n_captured = len(AttentionCapture.weights)
    print(f"\nCaptured attention from {n_captured} layers")

    if n_captured == 0:
        print("ERROR: No attention weights captured! "
              "The model might be using chunked prefill.")
        return

    attn_np = []
    for i, aw in enumerate(AttentionCapture.weights):
        a = np.array(aw)
        if i == 0:
            print(f"Attention shape per layer: {a.shape}")
        attn_np.append(a)

    # ── Stage 1 ──
    print("\n" + "=" * 50)
    print("STAGE 1: Modality Flow Heatmap")
    print("=" * 50)
    flows = stage1_modality_flow(
        attn_np, img_start, img_end, args.output_dir,
    )

    # ── Stage 2 ──
    print("\n" + "=" * 50)
    print("STAGE 2: Head Specialization Scatter")
    print("=" * 50)
    top_heads = stage2_head_scatter(
        attn_np, img_start, img_end,
        args.output_dir, args.threshold,
    )

    # ── Stage 3 ──
    print("\n" + "=" * 50)
    print("STAGE 3: Spatio-Temporal Overlay")
    print("=" * 50)
    stage3_spatiotemporal(
        attn_np, img_start, img_end,
        top_heads, args.video,
        (T, H, W),
        args.output_dir,
        spatial_merge_size=(
            model_config.vision_config.spatial_merge_size
        ),
    )

    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
