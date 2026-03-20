"""
Pixel masking representation analysis + attention heatmap visualization.

Runs 8 conditions on image samples to study position-vs-content:
  BL, PM-sink (paint sink patches black), PM-random, PM-content,
  PM-noise (Gaussian noise on sink patches), PM-shuffle (swap sink/content pixels),
  PM-all (black out all patches),
  SM-sink (attention blocking)

Generates 3 figure types:
  1. Attention heatmap grids (row1=input, row2=heatmap)
  2. phi scatter plots (phi_normal vs phi_masked)
  3. Sink persistence bar chart

Usage:
  TOKENIZERS_PARALLELISM=false uv run \
    --with "transformers<4.52" --with torch \
    --with datasets --with matplotlib --with scipy \
    python exp_pixel_masking_repr.py \
      --benchmarks mmstar \
      --max-per-benchmark 20 \
      --seed 42 \
      --output-dir ../../experiments/repr_analysis/pixel_masking
"""

import argparse
import gc
import json
import os
import time
from typing import Dict, List, Tuple

import mlx.core as mx
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ── Logger ───────────────────────────────────────────

class ExpLogger:
    def __init__(self, path: str):
        self.f = open(path, "a")

    def log(self, event: str, **kw):
        kw["event"] = event
        kw["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            kw["metal_mem_mb"] = round(
                mx.metal.get_active_memory() / 1e6, 1,
            )
        except Exception:
            pass
        self.f.write(json.dumps(kw, default=str) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


# ── Sink detection helpers ───────────────────────────

def detect_sinks(hs_np, s, e, sink_dims, layer, tau):
    """Detect sink tokens, return (sink_local, phi_vis)."""
    from sink_detect import rmsnorm
    hs = hs_np[layer]
    rms_val = np.abs(rmsnorm(hs))
    phi = np.max(
        np.stack([rms_val[:, d] for d in sink_dims], axis=-1),
        axis=-1,
    )
    vis_phi = phi[s:e]
    sink_local = np.where(vis_phi > tau)[0]
    return sink_local, vis_phi


# ── Forward with attention capture ───────────────────

def forward_capture_attn(
    model, ids, pv, attn_mask, extra,
    soft_mask_positions=None, seq_len=None,
):
    """
    Forward pass capturing hidden states AND attention weights.

    Returns (hs_np, attn_np, logits_np).
    attn_np[layer] has shape [n_heads, q_len, kv_len].
    """
    from mlx_vlm.models.cache import make_prompt_cache
    from sink_detect import CaptureStore
    from exp_three_way_masking import SoftMask

    CaptureStore.enable()

    if soft_mask_positions is not None and seq_len is not None:
        SoftMask.set_blocked(
            soft_mask_positions.tolist(), seq_len,
        )
    else:
        SoftMask.disable()

    model.language_model._position_ids = None
    model.language_model._rope_deltas = None

    cache = make_prompt_cache(model.language_model)
    eo = model.get_input_embeddings(
        ids, pv, mask=attn_mask, **extra,
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

    logits_np = np.array(out.logits[0, -1, :]).astype(np.float32)

    hs_np = []
    for h in CaptureStore.hidden_states:
        arr = np.array(h).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        hs_np.append(arr)

    attn_np = []
    for a in CaptureStore.attn_weights:
        arr = np.array(a).astype(np.float32)
        if arr.ndim == 4:
            arr = arr[0]  # [n_heads, q_len, kv_len]
        attn_np.append(arr)

    CaptureStore.disable()
    SoftMask.disable()
    del cache, eo, embeds, out
    mx.metal.clear_cache()

    return hs_np, attn_np, logits_np


def make_pixel_masked_pv(
    pv, mask_local_indices, grid_thw, sms=2,
):
    """Zero out patches in pixel_values for given local indices."""
    from exp_pixel_masking import sink_indices_to_pixel_mask

    T, H, W = grid_thw
    patch_idx = sink_indices_to_pixel_mask(
        mask_local_indices, (T, H, W), sms,
    )
    pv_np = np.array(pv).copy()
    valid = patch_idx[patch_idx < pv_np.shape[0]]
    if len(valid) > 0:
        pv_np[valid] = 0.0
    return mx.array(pv_np)


def make_noise_masked_pv(
    pv, mask_local_indices, grid_thw, rng, sms=2,
):
    """Replace patches at given indices with Gaussian noise."""
    from exp_pixel_masking import sink_indices_to_pixel_mask

    T, H, W = grid_thw
    patch_idx = sink_indices_to_pixel_mask(
        mask_local_indices, (T, H, W), sms,
    )
    pv_np = np.array(pv).copy()
    valid = patch_idx[patch_idx < pv_np.shape[0]]
    if len(valid) > 0:
        # Match mean/std of existing patch values
        mu = pv_np.mean()
        sigma = pv_np.std()
        noise = rng.normal(mu, sigma, pv_np[valid].shape)
        pv_np[valid] = noise.astype(pv_np.dtype)
    return mx.array(pv_np)


def make_shuffle_masked_pv(
    pv, sink_local, nonsink_local, grid_thw, rng, sms=2,
):
    """Swap pixel values between sink and content patches."""
    from exp_pixel_masking import sink_indices_to_pixel_mask

    T, H, W = grid_thw
    sink_patch_idx = sink_indices_to_pixel_mask(
        sink_local, (T, H, W), sms,
    )
    # Pick same number of content patches to swap with
    n_swap = min(len(sink_local), len(nonsink_local))
    swap_content = rng.choice(
        nonsink_local, size=n_swap, replace=False,
    )
    content_patch_idx = sink_indices_to_pixel_mask(
        swap_content, (T, H, W), sms,
    )

    pv_np = np.array(pv).copy()
    sink_valid = sink_patch_idx[sink_patch_idx < pv_np.shape[0]]
    cont_valid = content_patch_idx[
        content_patch_idx < pv_np.shape[0]
    ]
    n_swap_actual = min(len(sink_valid), len(cont_valid))
    if n_swap_actual > 0:
        sv = sink_valid[:n_swap_actual]
        cv = cont_valid[:n_swap_actual]
        pv_np[sv], pv_np[cv] = (
            pv_np[cv].copy(), pv_np[sv].copy(),
        )
    return mx.array(pv_np)


def make_all_black_pv(pv):
    """Zero out ALL patches."""
    pv_np = np.zeros_like(np.array(pv))
    return mx.array(pv_np)


# ── Attention heatmap visualization ──────────────────

def attn_to_spatial(
    attn_layer, s, e, grid_thw, sms=2,
    query_range=None, head_mode="mean",
    head_idx=None,
):
    """
    Convert attention weights to spatial heatmap.

    attn_layer: [n_heads, q_len, kv_len]
    Returns: [llm_H, llm_W] heatmap (averaged over frames if T>1)
    """
    T, H, W = grid_thw
    llm_H = H // sms
    llm_W = W // sms
    n_vis = e - s

    # Select query tokens (default: all non-visual tokens)
    if query_range is None:
        # Use last 10 text tokens as query
        q_end = attn_layer.shape[1]
        q_start = max(e, q_end - 10)
        query_range = slice(q_start, q_end)

    # For list-based indexing, extract submatrix first
    if isinstance(query_range, (list, np.ndarray)):
        q_idx = np.array(query_range)
        # Filter valid indices
        q_idx = q_idx[q_idx < attn_layer.shape[1]]
        if len(q_idx) == 0:
            return np.zeros((H // sms, W // sms)), head_idx
        attn_sub = attn_layer[:, q_idx, :][:, :, s:e]
    else:
        attn_sub = attn_layer[:, query_range, s:e]

    # Select heads
    if head_mode == "mean":
        attn_agg = attn_sub.mean(axis=(0, 1))  # [n_vis]
    elif head_mode == "single" and head_idx is not None:
        if isinstance(query_range, (list, np.ndarray)):
            attn_agg = attn_sub[head_idx].mean(axis=0)
        else:
            attn_agg = attn_layer[
                head_idx, query_range, s:e
            ].mean(axis=0)
    elif head_mode == "max_sink":
        head_means = attn_sub.mean(axis=(1, 2))
        best_head = np.argmax(head_means)
        attn_agg = attn_sub[best_head].mean(axis=0)
        head_idx = best_head
    else:
        attn_agg = attn_sub.mean(axis=(0, 1))

    # Reshape to spatial grid (average over frames)
    tokens_per_frame = llm_H * llm_W
    if n_vis >= tokens_per_frame:
        n_frames = n_vis // tokens_per_frame
        spatial = attn_agg[:n_frames * tokens_per_frame].reshape(
            n_frames, llm_H, llm_W,
        ).mean(axis=0)  # [llm_H, llm_W]
    else:
        # Fewer tokens than one frame — just reshape what we have
        side = int(np.sqrt(n_vis))
        spatial = attn_agg[:side*side].reshape(side, side)

    return spatial, head_idx


def get_sink_head_ranking(attn_layer, s, e, sink_abs):
    """Rank heads by attention to sink positions."""
    n_heads = attn_layer.shape[0]
    # Use text tokens as queries
    q_end = attn_layer.shape[1]
    q_start = max(e, q_end - 10)

    scores = []
    for hi in range(n_heads):
        attn_to_sinks = attn_layer[hi, q_start:q_end, sink_abs]
        attn_to_all_vis = attn_layer[hi, q_start:q_end, s:e]
        sink_frac = (
            attn_to_sinks.sum() / (attn_to_all_vis.sum() + 1e-12)
        )
        scores.append((hi, float(sink_frac)))

    scores.sort(key=lambda x: -x[1])
    return scores  # [(head_idx, sink_attention_fraction), ...]


def generate_heatmap_figure(
    image, attn_conditions, s, e, grid_thw, sms,
    sink_local, layers, output_path, sample_id="",
):
    """
    Generate the hero figure: input images + attention heatmaps.

    attn_conditions: dict of {cond_name: (attn_list, label, img)}
    """
    cond_names = list(attn_conditions.keys())
    n_conds = len(cond_names)
    n_layers = len(layers)

    fig, axes = plt.subplots(
        n_layers + 1, n_conds,
        figsize=(4 * n_conds, 3.5 * (n_layers + 1)),
    )
    if n_layers + 1 == 1:
        axes = axes.reshape(1, -1)

    # Row 0: Input images
    for ci, cname in enumerate(cond_names):
        _, label, img = attn_conditions[cname]
        ax = axes[0, ci]
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "(no image)", ha="center",
                    va="center", transform=ax.transAxes)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.axis("off")

    # Rows 1+: Attention heatmaps per layer
    for li_idx, layer_num in enumerate(layers):
        for ci, cname in enumerate(cond_names):
            attn_list, label, _ = attn_conditions[cname]
            ax = axes[li_idx + 1, ci]

            if layer_num < len(attn_list):
                spatial, _ = attn_to_spatial(
                    attn_list[layer_num], s, e,
                    grid_thw, sms,
                    head_mode="mean",
                )
                im = ax.imshow(
                    spatial, cmap="hot",
                    interpolation="nearest",
                )
                ax.set_title(f"L{layer_num}", fontsize=9)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center",
                        va="center", transform=ax.transAxes)
            ax.axis("off")

    fig.suptitle(
        f"Attention Heatmaps: BL vs Masking Conditions "
        f"(mean across heads) {sample_id}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_per_head_grid(
    attn_bl, s, e, grid_thw, sms, layer_num,
    output_path,
):
    """Generate 28-head grid for one layer (BL only)."""
    n_heads = attn_bl.shape[0]
    cols = 7
    rows = (n_heads + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(3 * cols, 3 * rows),
    )
    for hi in range(n_heads):
        r, c = divmod(hi, cols)
        ax = axes[r, c]
        spatial, _ = attn_to_spatial(
            attn_bl, s, e, grid_thw, sms,
            head_mode="single", head_idx=hi,
        )
        ax.imshow(spatial, cmap="hot", interpolation="nearest")
        ax.set_title(f"H{hi}", fontsize=8)
        ax.axis("off")

    # Clear unused subplots
    for hi in range(n_heads, rows * cols):
        r, c = divmod(hi, cols)
        axes[r, c].axis("off")

    fig.suptitle(
        f"All {n_heads} Heads — Layer {layer_num} (BL)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_sink_vs_content_heads(
    attn_conditions, s, e, grid_thw, sms,
    sink_abs, layer_num, output_path,
):
    """Show top-3 sink heads + top-1 content head for each condition."""
    cond_names = list(attn_conditions.keys())

    # Get head ranking from BL
    bl_attn = attn_conditions["BL"][0][layer_num]
    ranking = get_sink_head_ranking(bl_attn, s, e, sink_abs)
    top_sink_heads = [r[0] for r in ranking[:3]]
    content_head = ranking[-1][0]
    show_heads = top_sink_heads + [content_head]
    head_labels = [
        f"Sink H{h}" for h in top_sink_heads
    ] + [f"Content H{content_head}"]

    n_heads_show = len(show_heads)
    n_conds = len(cond_names)

    fig, axes = plt.subplots(
        n_heads_show, n_conds,
        figsize=(3.5 * n_conds, 3 * n_heads_show),
    )

    for hi_idx, (head_idx, hlabel) in enumerate(
        zip(show_heads, head_labels)
    ):
        for ci, cname in enumerate(cond_names):
            attn_list, label, _ = attn_conditions[cname]
            ax = axes[hi_idx, ci]

            if layer_num < len(attn_list):
                spatial, _ = attn_to_spatial(
                    attn_list[layer_num], s, e,
                    grid_thw, sms,
                    head_mode="single", head_idx=head_idx,
                )
                ax.imshow(
                    spatial, cmap="hot",
                    interpolation="nearest",
                )
            ax.axis("off")
            if hi_idx == 0:
                ax.set_title(label, fontsize=10,
                             fontweight="bold")
            if ci == 0:
                ax.set_ylabel(hlabel, fontsize=10)

    fig.suptitle(
        f"Sink vs Content Heads — Layer {layer_num}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ── Bidirectional attention flow ──────────────────────

def generate_bidirectional_flow(
    attn_conditions, s, e, grid_thw, sms,
    sink_local, layers, output_path, sample_id="",
):
    """
    Side-by-side: 'Who looks at sinks' vs 'Where sinks look'.

    Left column:  all non-visual queries → visual keys (sink-as-key)
    Right column: sink queries → visual keys (sink-as-query)
    """
    cond_names = list(attn_conditions.keys())
    n_conds = len(cond_names)
    n_layers = len(layers)

    fig, axes = plt.subplots(
        n_layers, n_conds * 2,
        figsize=(3 * n_conds * 2, 3 * n_layers),
    )
    if n_layers == 1:
        axes = axes.reshape(1, -1)

    sink_abs = sink_local + s

    for li_idx, layer_num in enumerate(layers):
        for ci, cname in enumerate(cond_names):
            attn_list, label, _ = attn_conditions[cname]
            if layer_num >= len(attn_list):
                continue
            attn_layer = attn_list[layer_num]
            q_end = attn_layer.shape[1]

            # ── Left: who looks at sinks (text → visual) ──
            ax_left = axes[li_idx, ci]
            q_start = max(e, q_end - 10)
            spatial_key, _ = attn_to_spatial(
                attn_layer, s, e, grid_thw, sms,
                query_range=slice(q_start, q_end),
                head_mode="mean",
            )
            ax_left.imshow(
                spatial_key, cmap="hot",
                interpolation="nearest",
            )
            if li_idx == 0:
                ax_left.set_title(
                    f"{label}\nSink-as-Key",
                    fontsize=8, fontweight="bold",
                )
            else:
                ax_left.set_title(f"L{layer_num}", fontsize=8)
            ax_left.axis("off")

            # ── Right: where sinks look (sink → visual) ──
            ax_right = axes[li_idx, n_conds + ci]
            if len(sink_abs) > 0:
                spatial_q, _ = attn_to_spatial(
                    attn_layer, s, e, grid_thw, sms,
                    query_range=sink_abs.tolist(),
                    head_mode="mean",
                )
                ax_right.imshow(
                    spatial_q, cmap="hot",
                    interpolation="nearest",
                )
            else:
                ax_right.text(
                    0.5, 0.5, "no sinks", ha="center",
                    va="center", transform=ax_right.transAxes,
                )
            if li_idx == 0:
                ax_right.set_title(
                    f"{label}\nSink-as-Query",
                    fontsize=8, fontweight="bold",
                )
            else:
                ax_right.set_title(f"L{layer_num}", fontsize=8)
            ax_right.axis("off")

    fig.suptitle(
        f"Bidirectional Attention Flow {sample_id}\n"
        f"Left half: Who looks at sinks | "
        f"Right half: Where sinks look",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def compute_attention_budget(
    attn_layer, s, e, sink_abs, nonsink_abs,
):
    """
    For sink queries, compute fraction of attention to each
    token group: system, sinks, content, text.

    Returns dict of {group: mean_fraction}.
    """
    if len(sink_abs) == 0:
        return {}

    n_heads = attn_layer.shape[0]
    q_end = attn_layer.shape[1]

    # Attention from sink queries, averaged over heads
    # attn_layer: [n_heads, q_len, kv_len]
    sink_attn = attn_layer[:, sink_abs, :]  # [H, n_sink, kv]
    sink_attn_mean = sink_attn.mean(axis=(0, 1))  # [kv_len]

    total = sink_attn_mean.sum() + 1e-12

    system_range = list(range(0, s))
    text_range = list(range(e, q_end))
    sink_list = sink_abs.tolist()
    content_list = nonsink_abs.tolist()

    budget = {
        "system": sink_attn_mean[system_range].sum() / total
        if system_range else 0.0,
        "sinks": sink_attn_mean[sink_list].sum() / total
        if sink_list else 0.0,
        "content": sink_attn_mean[content_list].sum() / total
        if content_list else 0.0,
        "text": sink_attn_mean[text_range].sum() / total
        if text_range else 0.0,
    }
    return {k: float(v) for k, v in budget.items()}


def generate_attention_budget_figure(
    budget_data, output_path,
):
    """
    Stacked bar chart: attention budget per layer per condition.

    budget_data: list of dicts with keys:
      layer, condition, system, sinks, content, text
    """
    import pandas as pd

    df = pd.DataFrame(budget_data)
    if df.empty:
        return

    layers = sorted(df["layer"].unique())
    conditions = sorted(
        df["condition"].unique(),
        key=lambda c: df[df["condition"] == c].index[0],
    )
    groups = ["system", "sinks", "content", "text"]
    colors = ["#9E9E9E", "#F44336", "#4CAF50", "#2196F3"]

    n_layers = len(layers)
    fig, axes = plt.subplots(
        1, n_layers, figsize=(4 * n_layers, 5),
        sharey=True,
    )
    if n_layers == 1:
        axes = [axes]

    for ax, layer_num in zip(axes, layers):
        sub = df[df["layer"] == layer_num]
        x = np.arange(len(conditions))
        bottom = np.zeros(len(conditions))

        for gi, (group, color) in enumerate(
            zip(groups, colors)
        ):
            vals = []
            for cond in conditions:
                row = sub[sub["condition"] == cond]
                if len(row) > 0:
                    vals.append(
                        row[group].values.mean()
                    )
                else:
                    vals.append(0)
            vals = np.array(vals)
            ax.bar(
                x, vals, bottom=bottom, width=0.6,
                color=color, label=group if ax == axes[0]
                else None,
            )
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(
            conditions, rotation=45, ha="right", fontsize=7,
        )
        ax.set_title(f"Layer {layer_num}", fontsize=10)
        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel("Fraction of sink attention")
    axes[0].legend(
        loc="upper left", fontsize=8,
        title="Token group",
    )
    fig.suptitle(
        "Attention Budget: Where Do Sinks Spend Their Attention?",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print("  Saved attention budget figure")


# ── phi scatter plot ─────────────────────────────────

def generate_phi_scatter(
    phi_data, output_dir, tau,
):
    """
    Generate phi scatter: phi_normal vs phi_masked.
    phi_data: list of dicts with keys:
      phi_bl, phi_pm_sink, phi_pm_random, phi_pm_content
    """
    configs = [
        ("phi_pm_sink", "PM-sink\n(sink patches black)",
         "#F44336"),
        ("phi_pm_random", "PM-random\n(random patches black)",
         "#FF9800"),
        ("phi_pm_content", "PM-content\n(non-sink patches black)",
         "#9C27B0"),
        ("phi_pm_noise", "PM-noise\n(sink patches noise)",
         "#4CAF50"),
        ("phi_pm_shuffle", "PM-shuffle\n(swap sink/content)",
         "#2196F3"),
        ("phi_pm_all", "PM-all\n(all patches black)",
         "#795548"),
    ]
    # Filter to configs that have data
    configs = [
        c for c in configs
        if any(c[0] in d for d in phi_data)
    ]
    n_cols = len(configs)
    fig, axes = plt.subplots(
        1, n_cols, figsize=(5 * n_cols, 5),
    )
    if n_cols == 1:
        axes = [axes]

    for ax, (key, title, color) in zip(axes, configs):
        all_bl, all_pm = [], []
        for d in phi_data:
            if key in d and "phi_bl" in d:
                n = min(len(d["phi_bl"]), len(d[key]))
                all_bl.extend(d["phi_bl"][:n].tolist())
                all_pm.extend(d[key][:n].tolist())

        if not all_bl:
            ax.text(0.5, 0.5, "No data", ha="center",
                    va="center", transform=ax.transAxes)
            continue

        all_bl = np.array(all_bl)
        all_pm = np.array(all_pm)

        ax.scatter(all_bl, all_pm, alpha=0.15, s=5,
                   color=color)
        # Diagonal reference
        lim = max(all_bl.max(), all_pm.max()) * 1.1
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3,
                label="y=x (position-driven)")
        # Threshold lines
        ax.axhline(y=tau, color="gray", linestyle=":",
                   alpha=0.5)
        ax.axvline(x=tau, color="gray", linestyle=":",
                   alpha=0.5)
        ax.set_xlabel("phi (normal)")
        ax.set_ylabel("phi (after masking)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        "phi Scatter: Position-Driven if Dots on Diagonal",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "fig_phi_scatter.png"),
        dpi=150,
    )
    plt.close(fig)
    print("  Saved fig_phi_scatter.png")


# ── Persistence bar chart ────────────────────────────

def generate_persistence_bar(
    persistence_data, output_dir,
):
    """
    Grouped bar: persistence rate + creation rate per condition.
    persistence_data: dict of {cond: (persist_rate, create_rate)}
    """
    conds = list(persistence_data.keys())
    persist = [persistence_data[c][0] for c in conds]
    create = [persistence_data[c][1] for c in conds]

    x = np.arange(len(conds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(
        x - width / 2, persist, width,
        label="Persistence (sink→sink)",
        color="#F44336", alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2, create, width,
        label="Creation (non-sink→sink)",
        color="#2196F3", alpha=0.8,
    )

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1%}", xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    ax.set_ylabel("Rate")
    ax.set_title("Sink Persistence & Creation After Pixel Masking")
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "fig_persistence_bar.png"),
        dpi=150,
    )
    plt.close(fig)
    print("  Saved fig_persistence_bar.png")


# ── Create masked input image for display ────────────

def create_masked_image(image, mask_local, grid_thw, sms=2):
    """
    Create a display image with masked patches shown as black.
    Returns PIL Image.
    """
    from PIL import Image as PILImage
    img = image.copy()
    img_w, img_h = img.size
    T, H, W = grid_thw
    llm_H = H // sms
    llm_W = W // sms

    # Each LLM token covers a rectangular patch of the image
    patch_h = img_h / llm_H
    patch_w = img_w / llm_W

    pixels = np.array(img)
    tokens_per_frame = llm_H * llm_W

    for idx in mask_local:
        frame = idx // tokens_per_frame
        if frame > 0:
            continue  # Only modify first frame for display
        local = idx % tokens_per_frame
        row = local // llm_W
        col = local % llm_W

        y0 = int(row * patch_h)
        y1 = int((row + 1) * patch_h)
        x0 = int(col * patch_w)
        x1 = int((col + 1) * patch_w)
        pixels[y0:y1, x0:x1] = 0  # Black

    return PILImage.fromarray(pixels)


# ── Main ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Pixel masking repr analysis + "
                    "attention heatmaps",
    )
    ap.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-VL-7B-Instruct-8bit",
    )
    ap.add_argument(
        "--benchmarks", nargs="+",
        default=["mmstar"],
        choices=["mmstar", "pope", "scienceqa"],
    )
    ap.add_argument(
        "--max-per-benchmark", type=int, default=20,
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tau", type=float, default=20.0)
    ap.add_argument("--detect-layer", type=int, default=14)
    ap.add_argument(
        "--heatmap-samples", type=int, default=3,
        help="Number of samples to generate full heatmap "
             "figures for",
    )
    ap.add_argument(
        "--output-dir",
        default="../../experiments/repr_analysis/pixel_masking",
    )
    ap.add_argument(
        "--discover-dims", action="store_true",
    )
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(
        os.path.join(args.output_dir, "heatmaps"), exist_ok=True,
    )

    sink_dims = [458, 2570]
    heatmap_layers = [3, 7, 14, 21, 27]

    # ── Init ───────────────────────────────────────────
    log_path = os.path.join(args.output_dir, "exp_run.jsonl")
    lg = ExpLogger(log_path)
    lg.log("exp_start", experiment="pixel_masking_repr",
           seed=args.seed, tau=args.tau)

    from mlx_vlm import load
    from exp_three_way_masking import (
        patch_model_v2, prepare_vision_input,
    )
    from sink_detect import (
        find_image_token_range, discover_sink_dims,
        CaptureStore, rmsnorm,
    )
    from benchmark_loaders import load_mmstar, load_pope

    print(f"Loading {args.model}...")
    model, processor = load(args.model)
    patch_model_v2(model)
    mcfg = model.config
    rng = np.random.RandomState(args.seed)

    def load_bm(name, max_s, seed):
        if name == "mmstar":
            return load_mmstar(max_s, seed)
        elif name == "pope":
            return load_pope(max_s, seed)
        else:
            raise ValueError(name)

    # ── Run ────────────────────────────────────────────
    phi_data = []  # For scatter plots
    persist_counts = {
        "PM-sink": [0, 0, 0, 0],
        "PM-random": [0, 0, 0, 0],
        "PM-content": [0, 0, 0, 0],
        "PM-noise": [0, 0, 0, 0],
        "PM-shuffle": [0, 0, 0, 0],
        "PM-all": [0, 0, 0, 0],
    }  # [sink_stayed, sink_lost, nonsink_stayed, nonsink_became]
    heatmap_count = 0
    budget_data = []  # For aggregate attention budget
    dims_discovered = False

    for bname in args.benchmarks:
        items = load_bm(
            bname, args.max_per_benchmark, args.seed,
        )
        print(f"\n{bname}: {len(items)} items")

        for qi, item in enumerate(items):
            try:
                t0 = time.time()

                image = item["image"]
                if image is None and "_load_image" in item:
                    image = item["_load_image"]()

                ids, pv, attn_mask, extra = prepare_vision_input(
                    processor, item["question"],
                    item["candidates"], image=image,
                )

                s, e = find_image_token_range(ids, mcfg)
                seq_len = ids.shape[1]
                n_vis = e - s

                # Get grid_thw
                grid_key = (
                    "image_grid_thw"
                    if "image_grid_thw" in extra
                    else "video_grid_thw"
                )
                if grid_key not in extra:
                    print(f"  [{qi}] SKIP: no grid_thw")
                    continue
                grid_thw = tuple(
                    int(x)
                    for x in np.array(extra[grid_key][0])
                )
                sms_cfg = getattr(mcfg, "vision_config", mcfg)
                sms = getattr(sms_cfg, "spatial_merge_size", 2)

                # ── BL ─────────────────────────────────
                hs_bl, attn_bl, logits_bl = (
                    forward_capture_attn(
                        model, ids, pv, attn_mask, extra,
                    )
                )

                # Auto-discover dims
                if args.discover_dims and not dims_discovered:
                    hs_3d = [
                        h[np.newaxis, :, :] for h in hs_bl
                    ]
                    img_dims, _ = discover_sink_dims(hs_3d)
                    if img_dims != sink_dims:
                        sink_dims = img_dims
                    dims_discovered = True

                # Detect sinks
                sink_local, phi_bl = detect_sinks(
                    hs_bl, s, e, sink_dims,
                    args.detect_layer, args.tau,
                )
                sink_abs = sink_local + s
                n_sink = len(sink_local)

                if n_sink == 0 or n_sink >= n_vis:
                    print(f"  [{qi}] SKIP: sink={n_sink}/{n_vis}")
                    continue

                # Non-sink indices
                all_vis_local = np.arange(n_vis)
                nonsink_local = np.array(sorted(
                    set(range(n_vis)) - set(sink_local.tolist())
                ))

                # Random selection (same count as sinks)
                rand_local = rng.choice(
                    all_vis_local, size=n_sink, replace=False,
                )

                # ── PM-sink ────────────────────────────
                pv_pm_sink = make_pixel_masked_pv(
                    pv, sink_local, grid_thw, sms,
                )
                hs_pms, attn_pms, _ = forward_capture_attn(
                    model, ids, pv_pm_sink, attn_mask, extra,
                )

                # ── PM-random ──────────────────────────
                pv_pm_rand = make_pixel_masked_pv(
                    pv, rand_local, grid_thw, sms,
                )
                hs_pmr, attn_pmr, _ = forward_capture_attn(
                    model, ids, pv_pm_rand, attn_mask, extra,
                )

                # ── PM-content ─────────────────────────
                pv_pm_cont = make_pixel_masked_pv(
                    pv, nonsink_local, grid_thw, sms,
                )
                hs_pmc, attn_pmc, _ = forward_capture_attn(
                    model, ids, pv_pm_cont, attn_mask, extra,
                )

                # ── PM-noise ──────────────────────────
                pv_pm_noise = make_noise_masked_pv(
                    pv, sink_local, grid_thw, rng, sms,
                )
                hs_pmn, attn_pmn, _ = forward_capture_attn(
                    model, ids, pv_pm_noise, attn_mask, extra,
                )

                # ── PM-shuffle ────────────────────────
                pv_pm_shuf = make_shuffle_masked_pv(
                    pv, sink_local, nonsink_local,
                    grid_thw, rng, sms,
                )
                hs_pmsh, attn_pmsh, _ = forward_capture_attn(
                    model, ids, pv_pm_shuf, attn_mask, extra,
                )

                # ── PM-all ────────────────────────────
                pv_pm_all = make_all_black_pv(pv)
                hs_pma, attn_pma, _ = forward_capture_attn(
                    model, ids, pv_pm_all, attn_mask, extra,
                )

                # ── SM-sink ────────────────────────────
                hs_sm, attn_sm, _ = forward_capture_attn(
                    model, ids, pv, attn_mask, extra,
                    soft_mask_positions=sink_abs,
                    seq_len=seq_len,
                )

                # ── Compute phi after masking ──────────
                _, phi_pms = detect_sinks(
                    hs_pms, s, e, sink_dims,
                    args.detect_layer, args.tau,
                )
                _, phi_pmr = detect_sinks(
                    hs_pmr, s, e, sink_dims,
                    args.detect_layer, args.tau,
                )
                _, phi_pmc = detect_sinks(
                    hs_pmc, s, e, sink_dims,
                    args.detect_layer, args.tau,
                )
                _, phi_pmn = detect_sinks(
                    hs_pmn, s, e, sink_dims,
                    args.detect_layer, args.tau,
                )
                _, phi_pmsh = detect_sinks(
                    hs_pmsh, s, e, sink_dims,
                    args.detect_layer, args.tau,
                )
                _, phi_pma = detect_sinks(
                    hs_pma, s, e, sink_dims,
                    args.detect_layer, args.tau,
                )

                phi_data.append({
                    "phi_bl": phi_bl,
                    "phi_pm_sink": phi_pms,
                    "phi_pm_random": phi_pmr,
                    "phi_pm_content": phi_pmc,
                    "phi_pm_noise": phi_pmn,
                    "phi_pm_shuffle": phi_pmsh,
                    "phi_pm_all": phi_pma,
                })

                # ── Persistence stats ──────────────────
                sink_set = set(sink_local.tolist())
                for pm_label, phi_pm in [
                    ("PM-sink", phi_pms),
                    ("PM-random", phi_pmr),
                    ("PM-content", phi_pmc),
                    ("PM-noise", phi_pmn),
                    ("PM-shuffle", phi_pmsh),
                    ("PM-all", phi_pma),
                ]:
                    new_sinks = set(
                        np.where(phi_pm > args.tau)[0].tolist()
                    )
                    ss = len(sink_set & new_sinks)
                    sl = len(sink_set - new_sinks)
                    ns = len(
                        (set(range(n_vis)) - sink_set)
                        - new_sinks
                    )
                    nb = len(
                        new_sinks - sink_set
                    )
                    persist_counts[pm_label][0] += ss
                    persist_counts[pm_label][1] += sl
                    persist_counts[pm_label][2] += ns
                    persist_counts[pm_label][3] += nb

                # ── Generate heatmap figures ───────────
                if heatmap_count < args.heatmap_samples:
                    # Create display images
                    img_pm_sink = create_masked_image(
                        image, sink_local, grid_thw, sms,
                    )
                    img_pm_rand = create_masked_image(
                        image, rand_local, grid_thw, sms,
                    )
                    img_pm_cont = create_masked_image(
                        image, nonsink_local, grid_thw, sms,
                    )

                    attn_conds = {
                        "BL": (attn_bl, "BL (baseline)", image),
                        "PM-sink": (attn_pms,
                                    "PM-sink (black)",
                                    img_pm_sink),
                        "PM-noise": (attn_pmn,
                                     "PM-noise (noise)",
                                     img_pm_sink),
                        "PM-shuffle": (attn_pmsh,
                                       "PM-shuffle (swap)",
                                       image),
                        "PM-all": (attn_pma,
                                   "PM-all (all black)",
                                   None),
                        "PM-random": (attn_pmr,
                                      "PM-random (black)",
                                      img_pm_rand),
                        "PM-content": (attn_pmc,
                                       "PM-content (black)",
                                       img_pm_cont),
                        "SM-sink": (attn_sm,
                                    "SM-sink (attn block)",
                                    image),
                    }

                    # Fig 1: Mean-head heatmaps
                    generate_heatmap_figure(
                        image, attn_conds, s, e,
                        grid_thw, sms, sink_local,
                        heatmap_layers,
                        os.path.join(
                            args.output_dir, "heatmaps",
                            f"heatmap_mean_{bname}_{qi:03d}.png",
                        ),
                        sample_id=f"[{bname} Q{qi}]",
                    )

                    # Fig: Sink vs content heads
                    for li in [14]:
                        if li < len(attn_bl):
                            generate_sink_vs_content_heads(
                                attn_conds, s, e,
                                grid_thw, sms, sink_abs, li,
                                os.path.join(
                                    args.output_dir, "heatmaps",
                                    f"heads_L{li}_{bname}"
                                    f"_{qi:03d}.png",
                                ),
                            )

                    # Fig: All 28 heads grid (BL only)
                    for li in [14]:
                        if li < len(attn_bl):
                            generate_per_head_grid(
                                attn_bl[li], s, e,
                                grid_thw, sms, li,
                                os.path.join(
                                    args.output_dir, "heatmaps",
                                    f"all_heads_L{li}_{bname}"
                                    f"_{qi:03d}.png",
                                ),
                            )

                    # Fig: Bidirectional attention flow
                    generate_bidirectional_flow(
                        attn_conds, s, e,
                        grid_thw, sms, sink_local,
                        heatmap_layers,
                        os.path.join(
                            args.output_dir, "heatmaps",
                            f"bidir_flow_{bname}_{qi:03d}.png",
                        ),
                        sample_id=f"[{bname} Q{qi}]",
                    )

                    heatmap_count += 1
                    print(f"    Heatmaps saved for Q{qi}")

                # ── Collect attention budget ───────────
                nonsink_abs = nonsink_local + s
                for layer_num in heatmap_layers:
                    for cond_label, attn_list in [
                        ("BL", attn_bl),
                        ("PM-sink", attn_pms),
                        ("PM-noise", attn_pmn),
                        ("PM-shuffle", attn_pmsh),
                        ("PM-all", attn_pma),
                        ("SM-sink", attn_sm),
                    ]:
                        if layer_num < len(attn_list):
                            b = compute_attention_budget(
                                attn_list[layer_num],
                                s, e, sink_abs, nonsink_abs,
                            )
                            if b:
                                b["layer"] = layer_num
                                b["condition"] = cond_label
                                budget_data.append(b)

                elapsed = time.time() - t0
                lg.log(
                    "sample", benchmark=bname, qi=qi,
                    n_vis=n_vis, n_sink=n_sink,
                    elapsed_sec=round(elapsed, 1),
                )
                print(
                    f"  [{qi+1}/{len(items)}] "
                    f"sink={n_sink}/{n_vis}  "
                    f"{elapsed:.1f}s"
                )

                del (hs_bl, attn_bl, hs_pms, attn_pms,
                     hs_pmr, attn_pmr, hs_pmc, attn_pmc,
                     hs_pmn, attn_pmn, hs_pmsh, attn_pmsh,
                     hs_pma, attn_pma, hs_sm, attn_sm)
                gc.collect()
                mx.metal.clear_cache()

            except Exception as ex:
                print(f"  ERROR Q{qi}: {ex}")
                lg.log("error", qi=qi, msg=str(ex))
                import traceback
                traceback.print_exc()

    # ── Generate aggregate figures ─────────────────────
    print(f"\nGenerating aggregate figures...")

    # phi scatter
    generate_phi_scatter(phi_data, args.output_dir, args.tau)

    # Persistence bar
    persist_rates = {}
    for pm_label, counts in persist_counts.items():
        ss, sl, ns, nb = counts
        persist = ss / (ss + sl) if (ss + sl) > 0 else 0
        create = nb / (ns + nb) if (ns + nb) > 0 else 0
        persist_rates[pm_label] = (persist, create)
        print(f"  {pm_label}: persist={persist:.1%}, "
              f"create={create:.1%}")

    generate_persistence_bar(persist_rates, args.output_dir)

    # Attention budget
    if budget_data:
        generate_attention_budget_figure(
            budget_data,
            os.path.join(
                args.output_dir, "fig_attention_budget.png",
            ),
        )

    lg.log("exp_end")
    lg.close()
    print(f"\nDone. Results -> {args.output_dir}")


if __name__ == "__main__":
    main()
