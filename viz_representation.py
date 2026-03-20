"""
Visualization for representation analysis experiments.

Generates 8 figures from JSONL logs and raw .npz files:
  1. SVD spectrum (log scale, representative layers)
  2. Layer-wise SVE evolution
  3. Layer-wise L2 norm progression
  4. Cosine similarity heatmap (layer × token position)
  5. Layer-wise mean cosine similarity
  6. Prediction entropy violin plot
  7. Logit KDE plot
  8. Top-K probability mass retention bar chart

Usage:
  uv run --with matplotlib --with scipy \
    python viz_representation.py \
      --input-dir ../../experiments/repr_analysis/full \
      --output-dir ../../experiments/repr_analysis/full/figures
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

COLORS = {
    "BL": "#2196F3",
    "SM-sink": "#F44336",
    "SM-random-90%": "#4CAF50",
}


def load_samples(jsonl_path: str) -> list:
    """Load all sample events from JSONL."""
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("event") == "sample":
                samples.append(rec)
    return samples


def load_raw(raw_dir: str) -> list:
    """Load raw .npz files."""
    raw_files = sorted(Path(raw_dir).glob("raw_*.npz"))
    return [np.load(f) for f in raw_files]


# ── Figure 1: SVD Spectrum ───────────────────────────

def fig_svd_spectrum(raw_data, output_dir):
    """SVD spectrum comparison for representative layers."""
    if not raw_data:
        print("  SKIP fig1: no raw data")
        return

    d = raw_data[0]
    layers = [0, 7, 14, 21, 27]
    available = [
        li for li in layers if f"svd_bl_L{li}" in d
    ]

    fig, axes = plt.subplots(
        1, len(available),
        figsize=(4 * len(available), 4),
        sharey=True,
    )
    if len(available) == 1:
        axes = [axes]

    for ax, li in zip(axes, available):
        svd_bl = d[f"svd_bl_L{li}"]
        svd_sm = d[f"svd_sm_L{li}"]
        svd_r90 = d[f"svd_r90_L{li}"]

        ax.semilogy(
            svd_bl, color=COLORS["BL"],
            label="BL", linewidth=1.5,
        )
        ax.semilogy(
            svd_sm, color=COLORS["SM-sink"],
            label="SM-sink", linewidth=1.5,
        )
        ax.semilogy(
            svd_r90, color=COLORS["SM-random-90%"],
            label="SM-rand-90%", linewidth=1.5,
        )
        ax.set_title(f"Layer {li}")
        ax.set_xlabel("Singular value index")
        if ax == axes[0]:
            ax.set_ylabel("Singular value (log scale)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Fig 1: Singular Value Spectrum (BL vs SM-sink "
        "vs SM-random-90%)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_svd_spectrum.png"))
    plt.close(fig)
    print("  Saved fig1_svd_spectrum.png")


# ── Figure 2: Layer-wise SVE ─────────────────────────

def fig_sve_by_layer(samples, output_dir):
    """Layer-wise singular value entropy, averaged."""
    if not samples:
        return

    n_layers = len(samples[0]["sve_bl"])
    layers = np.arange(n_layers)

    sve_bl = np.nanmean(
        [s["sve_bl"] for s in samples], axis=0,
    )
    sve_sm = np.nanmean(
        [s["sve_sm"] for s in samples], axis=0,
    )
    sve_r90 = np.nanmean(
        [s["sve_r90"] for s in samples], axis=0,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        layers, sve_bl, color=COLORS["BL"],
        marker="o", markersize=3,
        label="BL", linewidth=1.5,
    )
    ax.plot(
        layers, sve_sm, color=COLORS["SM-sink"],
        marker="s", markersize=3,
        label="SM-sink", linewidth=1.5,
    )
    ax.plot(
        layers, sve_r90, color=COLORS["SM-random-90%"],
        marker="^", markersize=3,
        label="SM-rand-90%", linewidth=1.5,
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Singular Value Entropy")
    ax.set_title(
        "Fig 2: Layer-wise SVE "
        f"(averaged over {len(samples)} samples)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_sve_by_layer.png"))
    plt.close(fig)
    print("  Saved fig2_sve_by_layer.png")


# ── Figure 3: Layer-wise L2 Norm ─────────────────────

def fig_norm_by_layer(samples, output_dir):
    """Layer-wise mean L2 norm, averaged."""
    if not samples:
        return

    n_layers = len(samples[0]["norm_bl"])
    layers = np.arange(n_layers)

    norm_bl = np.nanmean(
        [s["norm_bl"] for s in samples], axis=0,
    )
    norm_sm = np.nanmean(
        [s["norm_sm"] for s in samples], axis=0,
    )
    norm_r90 = np.nanmean(
        [s["norm_r90"] for s in samples], axis=0,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        layers, norm_bl, color=COLORS["BL"],
        marker="o", markersize=3,
        label="BL", linewidth=1.5,
    )
    ax.plot(
        layers, norm_sm, color=COLORS["SM-sink"],
        marker="s", markersize=3,
        label="SM-sink", linewidth=1.5,
    )
    ax.plot(
        layers, norm_r90, color=COLORS["SM-random-90%"],
        marker="^", markersize=3,
        label="SM-rand-90%", linewidth=1.5,
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean L2 Norm")
    ax.set_title(
        "Fig 3: Layer-wise Norm Progression "
        f"(averaged over {len(samples)} samples)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "fig3_norm_by_layer.png"),
    )
    plt.close(fig)
    print("  Saved fig3_norm_by_layer.png")


# ── Figure 4: Cosine Similarity Heatmap ──────────────

def fig_cosine_heatmap(raw_data, output_dir):
    """2D heatmap: layer × token position."""
    if not raw_data:
        print("  SKIP fig4: no raw data")
        return

    d = raw_data[0]
    cos_sm = d["cosine_per_token_sm"]   # [n_layers, seq_len]
    cos_r90 = d["cosine_per_token_r90"]

    fig, axes = plt.subplots(
        1, 2, figsize=(16, 6), sharey=True,
    )

    for ax, data, title in [
        (axes[0], cos_sm, "BL vs SM-sink"),
        (axes[1], cos_r90, "BL vs SM-random-90%"),
    ]:
        im = ax.imshow(
            data, aspect="auto", cmap="RdYlGn",
            vmin=0.5, vmax=1.0,
            origin="lower",
        )
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Layer")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="Cosine Similarity")

    fig.suptitle(
        "Fig 4: Cosine Similarity Heatmap "
        "(BL vs Masked Conditions)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "fig4_cosine_heatmap.png"),
    )
    plt.close(fig)
    print("  Saved fig4_cosine_heatmap.png")


# ── Figure 5: Layer-wise Mean Cosine ─────────────────

def fig_cosine_by_layer(samples, output_dir):
    """Layer-wise mean cosine similarity, averaged."""
    if not samples:
        return

    n_layers = len(samples[0]["cosine_bl_sm"])
    layers = np.arange(n_layers)

    cos_sm = np.nanmean(
        [s["cosine_bl_sm"] for s in samples], axis=0,
    )
    cos_r90 = np.nanmean(
        [s["cosine_bl_r90"] for s in samples], axis=0,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        layers, cos_sm, color=COLORS["SM-sink"],
        marker="s", markersize=3,
        label="BL vs SM-sink", linewidth=1.5,
    )
    ax.plot(
        layers, cos_r90, color=COLORS["SM-random-90%"],
        marker="^", markersize=3,
        label="BL vs SM-rand-90%", linewidth=1.5,
    )
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title(
        "Fig 5: Layer-wise Cosine Similarity Degradation "
        f"(averaged over {len(samples)} samples)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "fig5_cosine_by_layer.png"),
    )
    plt.close(fig)
    print("  Saved fig5_cosine_by_layer.png")


# ── Figure 6: Entropy Violin Plot ────────────────────

def fig_entropy_violin(samples, output_dir):
    """Violin plot of prediction entropy across conditions."""
    if not samples:
        return

    data = [
        [s["entropy_bl"] for s in samples
         if not np.isnan(s["entropy_bl"])],
        [s["entropy_sm"] for s in samples
         if not np.isnan(s["entropy_sm"])],
        [s["entropy_r90"] for s in samples
         if not np.isnan(s["entropy_r90"])],
    ]
    labels = ["BL", "SM-sink", "SM-rand-90%"]
    colors = [
        COLORS["BL"], COLORS["SM-sink"],
        COLORS["SM-random-90%"],
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    parts = ax.violinplot(
        data, positions=[1, 2, 3],
        showmeans=True, showmedians=True,
    )

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels)
    ax.set_ylabel("Prediction Entropy (bits)")
    ax.set_title(
        "Fig 6: Prediction Entropy Distribution "
        f"({len(samples)} samples)"
    )
    ax.grid(True, alpha=0.3, axis="y")

    # Add mean annotations
    for i, d in enumerate(data):
        mean_val = np.mean(d)
        ax.annotate(
            f"{mean_val:.2f}",
            xy=(i + 1, mean_val),
            xytext=(10, 5),
            textcoords="offset points",
            fontsize=9, color=colors[i],
        )

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "fig6_entropy_violin.png"),
    )
    plt.close(fig)
    print("  Saved fig6_entropy_violin.png")


# ── Figure 7: Logit KDE ─────────────────────────────

def fig_logit_kde(raw_data, output_dir):
    """KDE of logit values for representative samples."""
    if not raw_data:
        print("  SKIP fig7: no raw data")
        return

    n_plots = min(3, len(raw_data))
    fig, axes = plt.subplots(
        1, n_plots, figsize=(5 * n_plots, 4),
        sharey=True,
    )
    if n_plots == 1:
        axes = [axes]

    for idx, (ax, d) in enumerate(zip(axes, raw_data)):
        for key, label, color in [
            ("logits_bl", "BL", COLORS["BL"]),
            ("logits_sm", "SM-sink", COLORS["SM-sink"]),
            ("logits_r90", "SM-rand-90%",
             COLORS["SM-random-90%"]),
        ]:
            logits = d[key]
            # Use histogram as KDE approximation
            ax.hist(
                logits, bins=100, density=True,
                alpha=0.5, color=color, label=label,
                histtype="stepfilled",
            )
        ax.set_xlabel("Logit value")
        if ax == axes[0]:
            ax.set_ylabel("Density")
        ax.set_title(f"Sample {idx}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Fig 7: Logit Distribution (Full Vocabulary)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig7_logit_kde.png"))
    plt.close(fig)
    print("  Saved fig7_logit_kde.png")


# ── Figure 8: Top-K Retention Bar Chart ──────────────

def fig_topk_retention(samples, output_dir):
    """Bar chart of top-K probability mass retention."""
    if not samples:
        return

    ks = ["1", "5", "10"]

    retention_sm = {
        k: np.nanmean([
            s["topk_retention_sm"][k] for s in samples
        ]) for k in ks
    }
    retention_r90 = {
        k: np.nanmean([
            s["topk_retention_r90"][k] for s in samples
        ]) for k in ks
    }

    x = np.arange(len(ks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_sm = ax.bar(
        x - width / 2,
        [retention_sm[k] for k in ks],
        width, label="SM-sink",
        color=COLORS["SM-sink"], alpha=0.8,
    )
    bars_r90 = ax.bar(
        x + width / 2,
        [retention_r90[k] for k in ks],
        width, label="SM-rand-90%",
        color=COLORS["SM-random-90%"], alpha=0.8,
    )

    ax.set_xlabel("Top-K")
    ax.set_ylabel("Probability Mass Retention")
    ax.set_title(
        "Fig 8: Top-K Probability Mass Retention "
        f"(averaged over {len(samples)} samples)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top-{k}" for k in ks])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bars in [bars_sm, bars_r90]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "fig8_topk_retention.png"),
    )
    plt.close(fig)
    print("  Saved fig8_topk_retention.png")


# ── Summary Statistics ───────────────────────────────

def print_summary(samples):
    """Print summary statistics for the experiment."""
    n = len(samples)
    if n == 0:
        print("No samples to summarize.")
        return

    print(f"\n{'='*60}")
    print(f"SUMMARY ({n} samples)")
    print(f"{'='*60}")

    # KL divergence
    kl_sm = [s["kl_bl_sm"] for s in samples]
    kl_r90 = [s["kl_bl_r90"] for s in samples]
    print(f"\nKL Divergence (bits):")
    print(f"  BL vs SM-sink:     "
          f"mean={np.nanmean(kl_sm):.2f}  "
          f"median={np.nanmedian(kl_sm):.2f}  "
          f"std={np.nanstd(kl_sm):.2f}")
    print(f"  BL vs SM-rand-90%: "
          f"mean={np.nanmean(kl_r90):.2f}  "
          f"median={np.nanmedian(kl_r90):.2f}  "
          f"std={np.nanstd(kl_r90):.2f}")

    # Entropy
    ent_bl = [s["entropy_bl"] for s in samples]
    ent_sm = [s["entropy_sm"] for s in samples]
    ent_r90 = [s["entropy_r90"] for s in samples]
    print(f"\nPrediction Entropy (bits):")
    print(f"  BL:            mean={np.nanmean(ent_bl):.2f}")
    print(f"  SM-sink:       mean={np.nanmean(ent_sm):.2f}")
    print(f"  SM-rand-90%:   mean={np.nanmean(ent_r90):.2f}")

    # Top-1 retention
    top1_sm = [s["topk_retention_sm"]["1"] for s in samples]
    top1_r90 = [s["topk_retention_r90"]["1"] for s in samples]
    print(f"\nTop-1 Probability Retention:")
    print(f"  SM-sink:       mean={np.mean(top1_sm):.3f}")
    print(f"  SM-rand-90%:   mean={np.mean(top1_r90):.3f}")

    # Sink fraction
    sink_frac = [s["sink_frac"] for s in samples]
    print(f"\nSink Fraction: mean={np.mean(sink_frac):.1%}")

    # Cosine similarity at last layer
    cos_sm_last = [s["cosine_bl_sm"][-1] for s in samples]
    cos_r90_last = [s["cosine_bl_r90"][-1] for s in samples]
    print(f"\nCosine Similarity (last layer):")
    print(f"  BL vs SM-sink:     "
          f"mean={np.nanmean(cos_sm_last):.4f}")
    print(f"  BL vs SM-rand-90%: "
          f"mean={np.nanmean(cos_r90_last):.4f}")


# ── Main ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Visualize representation analysis results",
    )
    ap.add_argument(
        "--input-dir", required=True,
        help="Directory containing exp_run.jsonl and raw/",
    )
    ap.add_argument(
        "--output-dir", default=None,
        help="Output directory for figures (default: "
             "input_dir/figures)",
    )
    args = ap.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            args.input_dir, "figures",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    jsonl_path = os.path.join(args.input_dir, "exp_run.jsonl")
    print(f"Loading samples from {jsonl_path}...")
    samples = load_samples(jsonl_path)
    print(f"  Found {len(samples)} samples")

    raw_dir = os.path.join(args.input_dir, "raw")
    raw_data = []
    if os.path.isdir(raw_dir):
        raw_data = load_raw(raw_dir)
        print(f"  Found {len(raw_data)} raw files")

    # Print summary
    print_summary(samples)

    # Generate figures
    print(f"\nGenerating figures -> {args.output_dir}")
    fig_svd_spectrum(raw_data, args.output_dir)
    fig_sve_by_layer(samples, args.output_dir)
    fig_norm_by_layer(samples, args.output_dir)
    fig_cosine_heatmap(raw_data, args.output_dir)
    fig_cosine_by_layer(samples, args.output_dir)
    fig_entropy_violin(samples, args.output_dir)
    fig_logit_kde(raw_data, args.output_dir)
    fig_topk_retention(samples, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
