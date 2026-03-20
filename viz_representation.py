"""
Visualization for representation analysis experiments (v2, 7 conditions).

Generates figures from JSONL logs and raw .npz files.

Usage:
  uv run --with matplotlib --with scipy \
    python viz_representation.py \
      --input-dir ../../experiments/repr_analysis/v2_7cond \
      --output-dir ../../experiments/repr_analysis/v2_7cond/figures
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.figsize": (10, 6),
})

# 7 conditions with distinct colors
CONDS = ["sm", "he", "anti", "r50", "r90", "to"]
LABELS = {
    "bl": "BL",
    "sm": "SM-sink",
    "he": "HE",
    "anti": "SM-anti",
    "r50": "SM-rand-50%",
    "r90": "SM-rand-90%",
    "to": "Text-only",
}
COLORS = {
    "bl": "#2196F3",
    "sm": "#F44336",
    "he": "#E91E63",
    "anti": "#9C27B0",
    "r50": "#FF9800",
    "r90": "#4CAF50",
    "to": "#795548",
}


def load_samples(jsonl_path: str) -> list:
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("event") == "sample":
                samples.append(rec)
    return samples


def load_raw(raw_dir: str) -> list:
    raw_files = sorted(Path(raw_dir).glob("raw_*.npz"))
    return [np.load(f) for f in raw_files]


# ── Figure: Layer-wise Cosine Similarity (all conds) ──

def fig_cosine_by_layer(samples, output_dir):
    """Layer-wise cosine similarity for all conditions."""
    if not samples:
        return

    # Only plot conditions with same seq_len (not HE, TO)
    plot_conds = ["sm", "anti", "r50", "r90"]
    n_layers = len(samples[0].get("cosine_bl_sm", []))
    if n_layers == 0:
        return
    layers = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(12, 6))
    for c in plot_conds:
        key = f"cosine_bl_{c}"
        data = [s[key] for s in samples if key in s]
        if not data:
            continue
        mean_cos = np.nanmean(data, axis=0)
        ax.plot(
            layers, mean_cos, color=COLORS[c],
            marker="o", markersize=3,
            label=LABELS[c], linewidth=1.5,
        )
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title(
        "Layer-wise Cosine Similarity Degradation "
        f"(averaged over {len(samples)} samples)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0.7)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "fig_cosine_by_layer.png"),
    )
    plt.close(fig)
    print("  Saved fig_cosine_by_layer.png")


# ── Figure: Layer-wise L2 Norm (all conds) ───────────

def fig_norm_by_layer(samples, output_dir):
    """Layer-wise L2 norm for all conditions."""
    if not samples:
        return

    plot_conds = ["bl", "sm", "anti", "r50", "r90"]
    n_layers = len(samples[0].get("norm_bl", []))
    if n_layers == 0:
        return
    layers = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(12, 6))
    for c in plot_conds:
        key = f"norm_{c}"
        data = [s[key] for s in samples if key in s]
        if not data:
            continue
        mean_norm = np.nanmean(data, axis=0)
        ax.plot(
            layers, mean_norm, color=COLORS[c],
            marker="o", markersize=3,
            label=LABELS[c], linewidth=1.5,
        )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean L2 Norm")
    ax.set_title(
        "Layer-wise Norm Progression "
        f"(averaged over {len(samples)} samples)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "fig_norm_by_layer.png"),
    )
    plt.close(fig)
    print("  Saved fig_norm_by_layer.png")


# ── Figure: KL Divergence Bar Chart ──────────────────

def fig_kl_bar(samples, output_dir):
    """Bar chart of KL divergence across conditions."""
    if not samples:
        return

    conds = ["anti", "r50", "r90", "sm", "he", "to"]
    means, medians = [], []
    labels_list = []
    colors_list = []
    for c in conds:
        key = f"kl_bl_{c}"
        vals = [s[key] for s in samples if key in s]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            means.append(np.mean(vals))
            medians.append(np.median(vals))
            labels_list.append(LABELS[c])
            colors_list.append(COLORS[c])

    x = np.arange(len(labels_list))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        x, means, color=colors_list, alpha=0.8,
        edgecolor="white", linewidth=0.5,
    )
    # Add median markers
    ax.scatter(
        x, medians, color="black", zorder=5,
        s=30, marker="D", label="Median",
    )
    for i, (m, med) in enumerate(zip(means, medians)):
        ax.annotate(
            f"{m:.2f}", xy=(i, m), xytext=(0, 5),
            textcoords="offset points", ha="center",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels_list, rotation=15)
    ax.set_ylabel("KL Divergence (bits)")
    ax.set_title(
        f"KL Divergence: BL vs Each Condition "
        f"({len(samples)} samples)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_kl_bar.png"))
    plt.close(fig)
    print("  Saved fig_kl_bar.png")


# ── Figure: Entropy Bar Chart ────────────────────────

def fig_entropy_bar(samples, output_dir):
    """Bar chart of prediction entropy across conditions."""
    if not samples:
        return

    all_conds = ["bl"] + ["sm", "he", "anti", "r50", "r90", "to"]
    means = []
    labels_list = []
    colors_list = []
    for c in all_conds:
        key = f"entropy_{c}"
        vals = [s[key] for s in samples if key in s]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            means.append(np.mean(vals))
            labels_list.append(LABELS[c])
            colors_list.append(COLORS[c])

    x = np.arange(len(labels_list))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        x, means, color=colors_list, alpha=0.8,
        edgecolor="white", linewidth=0.5,
    )
    for i, m in enumerate(means):
        ax.annotate(
            f"{m:.2f}", xy=(i, m), xytext=(0, 5),
            textcoords="offset points", ha="center",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels_list, rotation=15)
    ax.set_ylabel("Prediction Entropy (bits)")
    ax.set_title(
        f"Prediction Entropy per Condition "
        f"({len(samples)} samples)"
    )
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "fig_entropy_bar.png"),
    )
    plt.close(fig)
    print("  Saved fig_entropy_bar.png")


# ── Figure: Cosine Similarity Heatmap ────────────────

def fig_cosine_heatmap(raw_data, output_dir):
    """2D heatmaps for SM-sink, SM-r50, SM-r90."""
    if not raw_data:
        print("  SKIP fig_cosine_heatmap: no raw data")
        return

    d = raw_data[0]
    panels = []
    for key, title in [
        ("cosine_per_token_sm", "BL vs SM-sink"),
        ("cosine_per_token_r50", "BL vs SM-rand-50%"),
        ("cosine_per_token_r90", "BL vs SM-rand-90%"),
    ]:
        if key in d:
            panels.append((d[key], title))

    if not panels:
        return

    fig, axes = plt.subplots(
        1, len(panels), figsize=(6 * len(panels), 5),
        sharey=True,
    )
    if len(panels) == 1:
        axes = [axes]

    for ax, (data, title) in zip(axes, panels):
        im = ax.imshow(
            data, aspect="auto", cmap="RdYlGn",
            vmin=0.5, vmax=1.0, origin="lower",
        )
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Layer")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="Cosine Similarity")

    fig.suptitle(
        "Cosine Similarity Heatmap (layer x token)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "fig_cosine_heatmap.png"),
    )
    plt.close(fig)
    print("  Saved fig_cosine_heatmap.png")


# ── Figure: Dose-Response Summary ────────────────────

def fig_dose_response(samples, output_dir):
    """Scatter: masking ratio vs cosine similarity / KL."""
    if not samples:
        return

    # For each sample, compute per-condition masking ratio
    # and corresponding cosine/KL
    points = []  # (ratio, cos, kl, condition)
    for s in samples:
        n_vis = s["n_vis"]
        n_sink = s["n_sink"]
        if n_vis == 0:
            continue

        configs = [
            ("anti", (n_vis - n_sink) / n_vis),
            ("r50", 0.5),
            ("r90", 0.9),
            ("sm", n_sink / n_vis),
        ]
        for c, ratio in configs:
            cos_key = f"cosine_bl_{c}"
            kl_key = f"kl_bl_{c}"
            if cos_key in s and kl_key in s:
                cos_val = s[cos_key][-1]
                kl_val = s[kl_key]
                if not np.isnan(cos_val) and not np.isnan(kl_val):
                    points.append((ratio, cos_val, kl_val, c))

    if not points:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for c in ["anti", "r50", "r90", "sm"]:
        pts = [(r, cos, kl) for r, cos, kl, cc in points
               if cc == c]
        if not pts:
            continue
        ratios = [p[0] for p in pts]
        cos_vals = [p[1] for p in pts]
        kl_vals = [p[2] for p in pts]

        ax1.scatter(
            ratios, cos_vals, color=COLORS[c],
            alpha=0.3, s=15, label=LABELS[c],
        )
        # Mean marker
        ax1.scatter(
            [np.mean(ratios)], [np.mean(cos_vals)],
            color=COLORS[c], s=100, marker="D",
            edgecolor="black", linewidth=1, zorder=5,
        )

        ax2.scatter(
            ratios, kl_vals, color=COLORS[c],
            alpha=0.3, s=15, label=LABELS[c],
        )
        ax2.scatter(
            [np.mean(ratios)], [np.mean(kl_vals)],
            color=COLORS[c], s=100, marker="D",
            edgecolor="black", linewidth=1, zorder=5,
        )

    ax1.set_xlabel("Masking Ratio")
    ax1.set_ylabel("Cosine Similarity (last layer)")
    ax1.set_title("Dose-Response: Masking Ratio vs Cosine")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Masking Ratio")
    ax2.set_ylabel("KL Divergence (bits)")
    ax2.set_title("Dose-Response: Masking Ratio vs KL")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Dose-Response: SM-sink causes MORE damage than "
        "equal-ratio random masking",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "fig_dose_response.png"),
    )
    plt.close(fig)
    print("  Saved fig_dose_response.png")


# ── Figure: Top-K Retention ──────────────────────────

def fig_topk_retention(samples, output_dir):
    """Top-K retention across all conditions."""
    if not samples:
        return

    conds = ["anti", "r50", "r90", "sm", "he", "to"]
    ks = ["1", "5", "10"]

    retention = {}
    for c in conds:
        key = f"topk_retention_{c}"
        for k in ks:
            vals = [
                s[key][k] for s in samples
                if key in s and k in s.get(key, {})
            ]
            vals = [v for v in vals if not np.isnan(v)]
            retention[(c, k)] = np.mean(vals) if vals else 0

    x = np.arange(len(ks))
    width = 0.12
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, c in enumerate(conds):
        vals = [retention[(c, k)] for k in ks]
        ax.bar(
            x + i * width, vals, width,
            label=LABELS[c], color=COLORS[c], alpha=0.8,
        )

    ax.set_xticks(x + width * (len(conds) - 1) / 2)
    ax.set_xticklabels([f"Top-{k}" for k in ks])
    ax.set_ylabel("Probability Mass Retention")
    ax.set_title(f"Top-K Retention ({len(samples)} samples)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "fig_topk_retention.png"),
    )
    plt.close(fig)
    print("  Saved fig_topk_retention.png")


# ── Summary ──────────────────────────────────────────

def print_summary(samples):
    n = len(samples)
    if n == 0:
        return

    print(f"\n{'='*60}")
    print(f"SUMMARY ({n} samples)")
    print(f"{'='*60}")

    print(f"\nSink fraction: "
          f"{np.mean([s['sink_frac'] for s in samples]):.1%}")

    conds_cos = ["sm", "anti", "r50", "r90"]
    conds_kl = ["sm", "he", "anti", "r50", "r90", "to"]

    print(f"\nCosine Similarity (last layer):")
    for c in conds_cos:
        key = f"cosine_bl_{c}"
        vals = [s[key][-1] for s in samples
                if key in s and not np.isnan(s[key][-1])]
        if vals:
            print(f"  {LABELS[c]:<14s}: {np.mean(vals):.4f}")

    print(f"\nKL Divergence (bits):")
    for c in conds_kl:
        key = f"kl_bl_{c}"
        vals = [s[key] for s in samples
                if key in s and not np.isnan(s[key])]
        if vals:
            print(f"  {LABELS[c]:<14s}: "
                  f"mean={np.mean(vals):.3f}  "
                  f"med={np.median(vals):.3f}")

    print(f"\nPrediction Entropy (bits):")
    bl_vals = [s["entropy_bl"] for s in samples
               if not np.isnan(s["entropy_bl"])]
    print(f"  {'BL':<14s}: {np.mean(bl_vals):.3f}")
    for c in conds_kl:
        key = f"entropy_{c}"
        vals = [s[key] for s in samples
                if key in s and not np.isnan(s[key])]
        if vals:
            print(f"  {LABELS[c]:<14s}: {np.mean(vals):.3f}")


# ── Main ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            args.input_dir, "figures",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    jsonl_path = os.path.join(args.input_dir, "exp_run.jsonl")
    print(f"Loading from {jsonl_path}...")
    samples = load_samples(jsonl_path)
    print(f"  {len(samples)} samples")

    raw_dir = os.path.join(args.input_dir, "raw")
    raw_data = []
    if os.path.isdir(raw_dir):
        raw_data = load_raw(raw_dir)
        print(f"  {len(raw_data)} raw files")

    print_summary(samples)

    print(f"\nGenerating figures -> {args.output_dir}")
    fig_cosine_by_layer(samples, args.output_dir)
    fig_norm_by_layer(samples, args.output_dir)
    fig_kl_bar(samples, args.output_dir)
    fig_entropy_bar(samples, args.output_dir)
    fig_cosine_heatmap(raw_data, args.output_dir)
    fig_dose_response(samples, args.output_dir)
    fig_topk_retention(samples, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
