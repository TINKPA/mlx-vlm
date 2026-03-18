"""
Visual Attention Sink detection on video for Qwen2.5-VL (MLX).

Reproduces the key analysis from "See What You Are Told" (ICLR 2025)
on video input:
  1. Discover sink dimensions via BOS token hidden states
  2. Detect sink tokens per layer (DimProspector logic)
  3. Visualize sink token spatial locations on video frames
  4. Compare attention to sink vs non-sink visual tokens

Usage:
  uv run --with "transformers<4.52" --with torch \
    --with seaborn --with matplotlib \
    python sink_detect.py \
      --model mlx-community/Qwen2.5-VL-7B-Instruct-8bit \
      --video test_video.mp4 \
      --prompt "Describe what happens in this video." \
      --output-dir sink_output
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image


# ── Capture store ─────────────────────────────────────────────

class CaptureStore:
    """Global store for hidden states AND attention weights."""
    hidden_states: List[mx.array] = []   # per layer, pre-attn
    attn_weights: List[mx.array] = []    # per layer
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


# ── Monkey-patches ────────────────────────────────────────────

_original_decoder_call = None
_original_attn_call = None


def _patched_decoder_layer_call(self, x, mask=None, cache=None,
                                position_ids=None):
    """Capture hidden state BEFORE attention in each layer."""
    if CaptureStore.enabled and x.shape[1] > 1:
        CaptureStore.hidden_states.append(x)
    return _original_decoder_call(
        self, x, mask=mask, cache=cache,
        position_ids=position_ids,
    )


def _patched_attention_call(self, x, mask=None, cache=None,
                            position_ids=None):
    """Compute attention manually and capture weights."""
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
        kv_seq_len += (
            cache.offset + 1 if cache is not None else 0
        )

    from mlx_vlm.models.qwen2_5_vl.language import (
        apply_multimodal_rotary_pos_emb,
    )
    cos, sin = self.rotary_emb(values, position_ids)

    if mask is not None and isinstance(mask, mx.array):
        mask = mask[..., :keys.shape[-2]]

    queries, keys = apply_multimodal_rotary_pos_emb(
        queries, keys, cos, sin, unqueeze_dim=1,
    )

    if cache is not None:
        keys, values = cache.update_and_fetch(keys, values)

    # GQA expand
    n_rep = self.n_heads // self.n_kv_heads
    if n_rep > 1:
        keys = mx.repeat(keys, n_rep, axis=1)
        values = mx.repeat(values, n_rep, axis=1)

    if CaptureStore.enabled and L > 1:
        attn_scores = (
            queries @ keys.transpose(0, 1, 3, 2)
        ) * self.scale

        if mask is not None and isinstance(mask, mx.array):
            attn_scores = attn_scores + mask
        elif mask is not None and mask == "causal":
            L_q = attn_scores.shape[2]
            L_k = attn_scores.shape[3]
            causal = mx.triu(
                mx.full((L_q, L_k), float("-inf")), k=1,
            )
            attn_scores = attn_scores + causal

        attn_w = mx.softmax(attn_scores, axis=-1)
        CaptureStore.attn_weights.append(attn_w)
        output = attn_w @ values
    else:
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values,
            scale=self.scale, mask=mask,
        )

    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(output)


def patch_model(model):
    """Monkey-patch to capture hidden states + attention."""
    global _original_decoder_call, _original_attn_call

    from mlx_vlm.models.qwen2_5_vl.language import (
        Attention, Qwen2VLDecoderLayer,
    )

    _original_decoder_call = Qwen2VLDecoderLayer.__call__
    Qwen2VLDecoderLayer.__call__ = (
        _patched_decoder_layer_call
    )

    _original_attn_call = Attention.__call__
    Attention.__call__ = _patched_attention_call

    n_layers = len(model.language_model.model.layers)
    print(f"Patched {n_layers} layers for hidden state "
          f"+ attention capture.")


# ── Token boundary detection ──────────────────────────────────

def find_image_token_range(input_ids, config):
    """Find [start, end) indices of image/video tokens."""
    ids = input_ids.flatten().tolist()
    img_id = config.image_token_id
    vid_id = config.video_token_id
    token_id = img_id if img_id in ids else vid_id
    positions = [i for i, t in enumerate(ids) if t == token_id]
    if not positions:
        raise ValueError("No image/video tokens found.")
    return positions[0], positions[-1] + 1


# ── Sink dimension discovery ──────────────────────────────────

def rmsnorm(hs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply RMSNorm (no learnable weights)."""
    variance = np.mean(hs ** 2, axis=-1, keepdims=True)
    return hs * np.reciprocal(np.sqrt(variance + eps))


def discover_sink_dims(
    hidden_states: List[np.ndarray],
    bos_idx: int = 0,
    top_k: int = 5,
) -> Tuple[List[int], np.ndarray]:
    """
    Discover sink dimensions by examining BOS token's
    hidden state activations across layers.

    Returns (sink_dims, activation_profile).
    """
    n_layers = len(hidden_states)
    hidden_dim = hidden_states[0].shape[-1]

    # Accumulate max |RMSNorm(h)| at BOS across layers
    bos_profile = np.zeros(hidden_dim)
    for li in range(2, n_layers):  # skip layers 0-1
        hs = hidden_states[li][0, bos_idx, :]  # [hidden_dim]
        rms = np.abs(rmsnorm(hs.reshape(1, -1)))[0]
        bos_profile = np.maximum(bos_profile, rms)

    # Top-k dimensions with highest activation
    top_dims = np.argsort(bos_profile)[::-1][:top_k]
    print(f"\nTop-{top_k} BOS activation dimensions:")
    for d in top_dims:
        print(f"  dim {d}: max |RMSNorm| = {bos_profile[d]:.2f}")

    # Sink dims = those with activation >> mean
    mean_act = np.mean(bos_profile)
    std_act = np.std(bos_profile)
    threshold = mean_act + 10 * std_act
    sink_dims = [
        int(d) for d in top_dims
        if bos_profile[d] > threshold
    ]
    if not sink_dims:
        # Fallback: use top-2
        sink_dims = [int(top_dims[0]), int(top_dims[1])]

    print(f"\nDiscovered sink dimensions: {sink_dims}")
    print(f"  (threshold = mean + 10*std "
          f"= {threshold:.2f})")
    return sink_dims, bos_profile


# ── Sink token detection (DimProspector) ──────────────────────

def detect_sink_tokens(
    hidden_states: List[np.ndarray],
    sink_dims: List[int],
    tau: float = 20.0,
    img_start: int = 0,
    img_end: int = 0,
) -> Dict[int, np.ndarray]:
    """
    Detect sink tokens per layer using DimProspector logic.
    Returns {layer: array of sink token indices}.
    """
    n_layers = len(hidden_states)
    sink_map = {}

    for li in range(2, n_layers):
        hs = hidden_states[li][0]  # [seq_len, hidden_dim]
        rms = np.abs(rmsnorm(hs))  # [seq_len, hidden_dim]
        # Max activation across sink dims
        phi = np.max(
            np.stack([rms[:, d] for d in sink_dims], axis=-1),
            axis=-1,
        )  # [seq_len]
        sink_indices = np.where(phi > tau)[0]
        sink_map[li] = sink_indices

    return sink_map


# ── Multi-layer φ aggregation ─────────────────────────────────

def score_phi_multilayer(
    capture_store_hs: List[np.ndarray],
    token_indices: Optional[slice] = None,
    layers: List[int] = None,
    weights: Optional[List[float]] = None,
    sink_dims: List[int] = None,
) -> np.ndarray:
    """
    Compute φ = max(|RMSNorm(h)|) on sink_dims, aggregated
    across multiple layers via weighted sum.

    Args:
        capture_store_hs: list of hidden states per layer,
            each shape [1, seq_len, hidden_dim] or
            [seq_len, hidden_dim].
        token_indices: optional slice to select tokens
            (default: all tokens).
        layers: which layers to use (default [14,18,22,24]).
        weights: per-layer weights for aggregation
            (default: equal weights, normalized to sum=1).
        sink_dims: dimensions to check (default [458, 2570]).

    Returns:
        np.ndarray of shape [n_tokens] with aggregated φ.
    """
    if layers is None:
        layers = [14, 18, 22, 24]
    if sink_dims is None:
        sink_dims = [458, 2570]
    if weights is None:
        weights = [1.0 / len(layers)] * len(layers)
    else:
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]

    n_hs = len(capture_store_hs)
    aggregated = None

    for li, w in zip(layers, weights):
        idx = min(li, n_hs - 1)
        h = capture_store_hs[idx]
        if isinstance(h, mx.array):
            h = np.array(h)
        if h.ndim == 3:
            h = h[0]  # [seq_len, hidden_dim]
        if token_indices is not None:
            h = h[token_indices]
        rms = np.abs(rmsnorm(h))
        phi = np.max(
            np.stack([rms[:, d] for d in sink_dims], axis=-1),
            axis=-1,
        )  # [n_tokens]
        if aggregated is None:
            aggregated = w * phi
        else:
            aggregated = aggregated + w * phi

    return aggregated


# ── Visualization functions ───────────────────────────────────

def plot_phi_profile(
    hidden_states: List[np.ndarray],
    sink_dims: List[int],
    img_start: int,
    img_end: int,
    output_dir: str,
    sample_layers: List[int] = None,
):
    """
    Plot phi (sink dimension activation) for all tokens,
    highlighting BOS, visual sink, and visual non-sink.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n_layers = len(hidden_states)
    if sample_layers is None:
        sample_layers = [2, 5, 10, 15, 20, 25]
        sample_layers = [l for l in sample_layers if l < n_layers]

    fig, axes = plt.subplots(
        len(sample_layers), 1,
        figsize=(14, 3 * len(sample_layers)),
        sharex=False,
    )
    if len(sample_layers) == 1:
        axes = [axes]

    for ax_i, li in enumerate(sample_layers):
        hs = hidden_states[li][0]  # [seq, dim]
        rms = np.abs(rmsnorm(hs))
        phi = np.max(
            np.stack([rms[:, d] for d in sink_dims], axis=-1),
            axis=-1,
        )

        seq_len = phi.shape[0]
        x = np.arange(seq_len)

        # Color by modality
        colors = np.full(seq_len, 0.7)  # grey for text
        colors[img_start:img_end] = 0.3  # blue for visual
        colors[0] = 1.0  # BOS

        # Scatter
        ax = axes[ax_i]
        vis_phi = phi[img_start:img_end]
        txt_phi = np.concatenate(
            [phi[:img_start], phi[img_end:]]
        )

        ax.scatter(
            x[img_start:img_end], vis_phi,
            c="steelblue", s=8, alpha=0.5,
            label="visual tokens",
        )
        txt_x = np.concatenate(
            [x[:img_start], x[img_end:]]
        )
        ax.scatter(
            txt_x, txt_phi,
            c="grey", s=8, alpha=0.3,
            label="text tokens",
        )
        ax.scatter(
            [0], [phi[0]],
            c="red", s=40, zorder=5,
            label=f"BOS (φ={phi[0]:.1f})",
        )

        # Highlight sink tokens (phi > tau)
        sink_mask = phi > 20
        if sink_mask.any():
            ax.scatter(
                x[sink_mask], phi[sink_mask],
                facecolors="none", edgecolors="red",
                s=50, linewidths=1.5, zorder=4,
                label=f"sink (φ>20, n={sink_mask.sum()})",
            )

        ax.axhline(y=20, color="red", linestyle="--",
                    alpha=0.4, linewidth=1)
        ax.axvspan(img_start, img_end, alpha=0.05,
                   color="blue")
        ax.set_ylabel("φ (sink dim value)")
        ax.set_title(f"Layer {li}")
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Token Index")
    fig.suptitle(
        "Sink Dimension Activation (φ) per Layer\n"
        f"sink dims = {sink_dims}, τ = 20",
        fontsize=14,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "phi_profile.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


def plot_sink_spatial_map(
    hidden_states: List[np.ndarray],
    sink_dims: List[int],
    img_start: int,
    img_end: int,
    grid_thw: Tuple[int, int, int],
    spatial_merge_size: int,
    video_path: str,
    output_dir: str,
    sample_layers: List[int] = None,
):
    """
    Map sink tokens to their spatial locations on video
    frames. Shows which pixel regions are sinks.
    """
    import matplotlib.pyplot as plt

    T, H, W = grid_thw
    llm_H = H // spatial_merge_size
    llm_W = W // spatial_merge_size
    n_tokens_per_frame = llm_H * llm_W
    n_vis = img_end - img_start
    n_frames = n_vis // n_tokens_per_frame

    n_layers = len(hidden_states)
    if sample_layers is None:
        sample_layers = [5, 10, 15, 20, 25]
        sample_layers = [l for l in sample_layers if l < n_layers]

    # Extract actual video frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, total_frames // n_frames)
    frames_rgb = []
    for fi in range(min(n_frames, 8)):
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            min(fi * frame_step, total_frames - 1),
        )
        ret, frame = cap.read()
        if ret:
            frames_rgb.append(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
    cap.release()

    n_show_frames = min(len(frames_rgb), 6)

    fig, axes = plt.subplots(
        len(sample_layers), n_show_frames,
        figsize=(3 * n_show_frames, 3 * len(sample_layers)),
    )
    if len(sample_layers) == 1:
        axes = axes.reshape(1, -1)
    if n_show_frames == 1:
        axes = axes.reshape(-1, 1)

    for row, li in enumerate(sample_layers):
        hs = hidden_states[li][0]
        rms = np.abs(rmsnorm(hs))
        phi = np.max(
            np.stack([rms[:, d] for d in sink_dims], axis=-1),
            axis=-1,
        )
        vis_phi = phi[img_start:img_end]

        for col in range(n_show_frames):
            frame_phi = vis_phi[
                col * n_tokens_per_frame:
                (col + 1) * n_tokens_per_frame
            ]
            if len(frame_phi) != n_tokens_per_frame:
                axes[row, col].axis("off")
                continue

            # Reshape to spatial grid
            phi_map = frame_phi.reshape(llm_H, llm_W)

            # Create binary sink mask
            sink_mask = (phi_map > 20).astype(np.float32)
            n_sink = int(sink_mask.sum())
            n_total = llm_H * llm_W

            # Overlay on frame
            frame = frames_rgb[col]
            h_f, w_f = frame.shape[:2]

            # Upsample phi map for overlay
            phi_up = cv2.resize(
                phi_map.astype(np.float32),
                (w_f, h_f),
                interpolation=cv2.INTER_NEAREST,
            )
            # Normalize phi to [0, 1] for color
            phi_norm = np.clip(phi_up / 60.0, 0, 1)
            heatmap = cv2.applyColorMap(
                (phi_norm * 255).astype(np.uint8),
                cv2.COLORMAP_HOT,
            )
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            overlay = (
                0.5 * frame.astype(np.float32)
                + 0.5 * heatmap.astype(np.float32)
            ).astype(np.uint8)

            axes[row, col].imshow(overlay)
            axes[row, col].set_title(
                f"F{col} sink={n_sink}/{n_total}",
                fontsize=9,
            )
            axes[row, col].axis("off")

        axes[row, 0].set_ylabel(
            f"L{li}", fontsize=12, rotation=0,
            labelpad=30, va="center",
        )

    fig.suptitle(
        "Sink Token Spatial Map (φ > τ=20)\n"
        "bright = high sink activation",
        fontsize=14,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "sink_spatial_map.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


def plot_sink_attention_comparison(
    attn_weights: List[np.ndarray],
    hidden_states: List[np.ndarray],
    sink_dims: List[int],
    img_start: int,
    img_end: int,
    output_dir: str,
):
    """
    Compare attention TO sink visual tokens vs non-sink
    visual tokens across layers. Core claim of the paper.
    """
    import matplotlib.pyplot as plt

    n_layers = min(len(attn_weights), len(hidden_states))
    layers = list(range(2, n_layers))

    attn_to_sink = []
    attn_to_nonsink = []
    n_sink_per_layer = []

    for li in layers:
        # Detect sink tokens at this layer
        hs = hidden_states[li][0]
        rms = np.abs(rmsnorm(hs))
        phi = np.max(
            np.stack([rms[:, d] for d in sink_dims], axis=-1),
            axis=-1,
        )
        vis_phi = phi[img_start:img_end]
        sink_local = np.where(vis_phi > 20)[0]
        nonsink_local = np.where(vis_phi <= 20)[0]

        n_sink_per_layer.append(len(sink_local))

        if len(sink_local) == 0 or len(nonsink_local) == 0:
            attn_to_sink.append(0)
            attn_to_nonsink.append(0)
            continue

        # Attention from text tokens → visual tokens
        # attn: [1, n_heads, Q, K]
        aw = attn_weights[li][0]  # [n_heads, Q, K]
        # Average over heads
        aw_mean = aw.mean(axis=0)  # [Q, K]

        # Text query tokens
        seq_len = aw_mean.shape[0]
        txt_mask = np.ones(seq_len, dtype=bool)
        txt_mask[img_start:img_end] = False
        txt_rows = aw_mean[txt_mask]  # [n_txt, K]

        # Avg attention to sink vs nonsink visual tokens
        sink_abs = sink_local + img_start
        nonsink_abs = nonsink_local + img_start

        avg_sink = txt_rows[:, sink_abs].mean()
        avg_nonsink = txt_rows[:, nonsink_abs].mean()

        attn_to_sink.append(float(avg_sink))
        attn_to_nonsink.append(float(avg_nonsink))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Attention comparison
    ax1.plot(layers, attn_to_sink, "r-o",
             markersize=4, label="txt→vis_SINK")
    ax1.plot(layers, attn_to_nonsink, "b-o",
             markersize=4, label="txt→vis_nonSINK")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Avg Attention Weight")
    ax1.set_title(
        "Text → Visual Token Attention: "
        "Sink vs Non-Sink"
    )
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Number of sink tokens
    ax2.bar(layers, n_sink_per_layer, color="coral",
            alpha=0.7)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("# Sink Tokens")
    ax2.set_title(
        f"Number of Visual Sink Tokens per Layer "
        f"(out of {img_end - img_start} visual tokens)"
    )
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(
        output_dir, "sink_attn_comparison.png",
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


def plot_sink_consistency(
    hidden_states: List[np.ndarray],
    sink_dims: List[int],
    img_start: int,
    img_end: int,
    grid_thw: Tuple[int, int, int],
    spatial_merge_size: int,
    output_dir: str,
):
    """
    Check if sink tokens occupy the same spatial positions
    across frames (key prediction from the paper).
    """
    import matplotlib.pyplot as plt

    T, H, W = grid_thw
    llm_H = H // spatial_merge_size
    llm_W = W // spatial_merge_size
    n_per_frame = llm_H * llm_W
    n_vis = img_end - img_start
    n_frames = n_vis // n_per_frame

    # Use a representative middle layer
    n_layers = len(hidden_states)
    test_layer = n_layers // 2

    hs = hidden_states[test_layer][0]
    rms = np.abs(rmsnorm(hs))
    phi = np.max(
        np.stack([rms[:, d] for d in sink_dims], axis=-1),
        axis=-1,
    )
    vis_phi = phi[img_start:img_end]

    # Reshape to [n_frames, H, W]
    sink_maps = []
    for fi in range(n_frames):
        frame_phi = vis_phi[
            fi * n_per_frame:(fi + 1) * n_per_frame
        ]
        if len(frame_phi) == n_per_frame:
            sink_maps.append(
                (frame_phi > 20).astype(np.float32).reshape(
                    llm_H, llm_W,
                )
            )

    if len(sink_maps) < 2:
        print("Not enough frames for consistency analysis.")
        return

    # Compute pairwise IoU of sink positions
    n = len(sink_maps)
    iou_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            inter = (sink_maps[i] * sink_maps[j]).sum()
            union = np.clip(
                sink_maps[i] + sink_maps[j], 0, 1,
            ).sum()
            iou_matrix[i, j] = (
                inter / union if union > 0 else 0
            )

    # Also compute cumulative sink frequency map
    freq_map = np.mean(sink_maps, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # IoU heatmap
    import matplotlib.pyplot as plt
    im = ax1.imshow(iou_matrix, cmap="YlOrRd",
                    vmin=0, vmax=1)
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Frame")
    ax1.set_title(
        f"Sink Position IoU (Layer {test_layer})\n"
        f"mean IoU = {iou_matrix.mean():.3f}"
    )
    fig.colorbar(im, ax=ax1)

    # Frequency map
    im2 = ax2.imshow(freq_map, cmap="hot", vmin=0, vmax=1)
    ax2.set_title(
        f"Sink Frequency Map (Layer {test_layer})\n"
        f"bright = sink in most frames"
    )
    ax2.set_xlabel(f"{llm_W} spatial columns")
    ax2.set_ylabel(f"{llm_H} spatial rows")
    fig.colorbar(im2, ax=ax2)

    plt.tight_layout()
    path = os.path.join(
        output_dir, "sink_consistency.png",
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")

    # Print summary stats
    off_diag = iou_matrix[
        ~np.eye(n, dtype=bool)
    ]
    print(f"\nSink Consistency (Layer {test_layer}):")
    print(f"  Mean IoU across frames: {off_diag.mean():.3f}")
    print(f"  Min IoU: {off_diag.min():.3f}")
    print(f"  Max IoU: {off_diag.max():.3f}")
    if off_diag.mean() > 0.5:
        print("  → HIGH consistency: sink positions are "
              "largely fixed across frames (matches paper)")
    elif off_diag.mean() > 0.2:
        print("  → MODERATE consistency: some positional "
              "overlap but frames differ")
    else:
        print("  → LOW consistency: sink positions vary "
              "across frames (differs from paper)")


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visual Attention Sink detection on video"
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
    parser.add_argument("--output-dir", default="sink_output")
    parser.add_argument(
        "--tau", type=float, default=20.0,
        help="Sink detection threshold (paper default: 20)",
    )
    parser.add_argument(
        "--paper-dims", type=int, nargs="+",
        default=None,
        help="Override sink dims (e.g. --paper-dims 458 2570)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ──
    from mlx_vlm import load, generate
    from mlx_vlm.utils import load_config
    from mlx_vlm.video_generate import process_vision_info

    print(f"Loading model: {args.model}")
    model, processor = load(args.model)
    config = load_config(args.model)

    # ── Patch for capture ──
    patch_model(model)

    # ── Prepare video input ──
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

    T, H, W = (
        int(grid_thw_np[0]),
        int(grid_thw_np[1]),
        int(grid_thw_np[2]),
    )
    print(f"Video grid: T={T}, H={H}, W={W}")

    model_config = model.config
    img_start, img_end = find_image_token_range(
        input_ids, model_config,
    )
    n_vis = img_end - img_start
    print(f"Image tokens: [{img_start}, {img_end}) "
          f"= {n_vis} tokens")
    print(f"Total sequence length: {input_ids.shape[1]}")

    # ── Run prefill with capture ──
    print("\nRunning prefill with hidden state "
          "& attention capture...")
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
    print(f"Model output: {output.text}")
    CaptureStore.disable()

    # ── Convert to numpy ──
    n_hs = len(CaptureStore.hidden_states)
    n_aw = len(CaptureStore.attn_weights)
    print(f"\nCaptured {n_hs} hidden states, "
          f"{n_aw} attention weights")

    if n_hs == 0:
        print("ERROR: No hidden states captured!")
        return

    hs_np = [np.array(h) for h in CaptureStore.hidden_states]
    aw_np = [np.array(a) for a in CaptureStore.attn_weights]

    hidden_dim = hs_np[0].shape[-1]
    print(f"Hidden dim: {hidden_dim}")

    # ── Step 1: Discover sink dimensions ──
    print("\n" + "=" * 50)
    print("STEP 1: Discovering Sink Dimensions")
    print("=" * 50)

    if args.paper_dims:
        sink_dims = args.paper_dims
        print(f"Using paper-specified dims: {sink_dims}")
    else:
        sink_dims, bos_profile = discover_sink_dims(hs_np)

        # Compare with paper's Qwen2-VL-7B dims
        paper_dims = [458, 2570]
        print(f"\nPaper's Qwen2-VL-7B dims: {paper_dims}")
        for d in paper_dims:
            if d < len(bos_profile):
                print(
                    f"  dim {d}: "
                    f"|RMSNorm| = {bos_profile[d]:.2f}"
                )
        match = set(sink_dims) & set(paper_dims)
        if match:
            print(f"  → MATCH on dims: {match}")
        else:
            print("  → NO MATCH (Qwen2.5-VL may differ "
                  "from Qwen2-VL)")

    # ── Step 2: Detect sink tokens ──
    print("\n" + "=" * 50)
    print("STEP 2: Detecting Sink Tokens")
    print("=" * 50)

    sink_map = detect_sink_tokens(
        hs_np, sink_dims, args.tau, img_start, img_end,
    )
    for li in sorted(sink_map.keys()):
        vis_sinks = [
            s for s in sink_map[li]
            if img_start <= s < img_end
        ]
        if vis_sinks:
            print(f"  Layer {li:2d}: {len(vis_sinks)} "
                  f"visual sink tokens "
                  f"(of {n_vis} total)")

    # ── Step 3: Visualizations ──
    print("\n" + "=" * 50)
    print("STEP 3: Generating Visualizations")
    print("=" * 50)

    spatial_merge = (
        model_config.vision_config.spatial_merge_size
    )

    plot_phi_profile(
        hs_np, sink_dims, img_start, img_end,
        args.output_dir,
    )

    plot_sink_spatial_map(
        hs_np, sink_dims, img_start, img_end,
        (T, H, W), spatial_merge,
        args.video, args.output_dir,
    )

    if aw_np:
        plot_sink_attention_comparison(
            aw_np, hs_np, sink_dims,
            img_start, img_end, args.output_dir,
        )

    plot_sink_consistency(
        hs_np, sink_dims, img_start, img_end,
        (T, H, W), spatial_merge, args.output_dir,
    )

    print(f"\nAll outputs saved to {args.output_dir}/")
    print("\nSummary of outputs:")
    print("  phi_profile.png        — φ activation per token")
    print("  sink_spatial_map.png   — sink locations on frames")
    print("  sink_attn_comparison.png — attn to sink vs non")
    print("  sink_consistency.png   — cross-frame IoU")


if __name__ == "__main__":
    main()
