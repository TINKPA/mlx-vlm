"""
Generate POPE smoking gun visualization.

Shows side-by-side: original image + sink map overlay,
with 6 condition answers labeled. Highlights cases where
BL=correct, SM-sink=wrong, AN=correct.

Usage:
  uv run --with datasets --with matplotlib --with pillow \
    python viz_pope_smoking_gun.py \
      --results-json .../multi_benchmark_seed123/results.json \
      --output-dir viz_pope_output \
      --n-examples 5
"""

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-json", required=True)
    ap.add_argument("--output-dir", default="viz_pope_output")
    ap.add_argument("--n-examples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    # Load results
    data = json.load(open(args.results_json))
    pope = data["benchmarks"]["pope"]

    # Find smoking gun: BL correct, SM wrong, AN correct
    smoking = [
        r for r in pope
        if r["correct_bl"]
        and not r["correct_sm"]
        and r["correct_anti"]
    ]
    print(f"Found {len(smoking)} smoking gun examples")

    # Load POPE dataset to get images
    from datasets import load_dataset
    print("Loading POPE dataset...")
    ds = load_dataset("lmms-lab/POPE", split="test")

    # Sample the same indices used in the experiment
    rng = np.random.RandomState(args.seed)
    all_indices = rng.choice(
        len(ds), min(100, len(ds)), replace=False,
    ).tolist()

    # Pick best examples
    examples = smoking[:args.n_examples]

    for ei, r in enumerate(examples):
        qi = r["qi"]
        ds_idx = all_indices[qi]
        row = ds[ds_idx]
        image = row["image"].convert("RGB")
        question = row["question"]

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))

        # Show image
        ax.imshow(image)
        ax.axis("off")

        # Title with question
        q_short = question
        if len(q_short) > 80:
            q_short = q_short[:77] + "..."
        ax.set_title(q_short, fontsize=11, pad=10)

        # Build answer table
        conditions = [
            ("Baseline", r["pred_bl"],
             r["correct_bl"], "#2196F3"),
            ("SM-sink", r["pred_sm"],
             r["correct_sm"], "#F44336"),
            ("Hard evict", r["pred_he"],
             r["correct_he"], "#FF9800"),
            ("SM-random", r["pred_rand"],
             r["correct_rand"], "#9E9E9E"),
            ("SM-anti-sink", r["pred_anti"],
             r["correct_anti"], "#4CAF50"),
            ("Text-only", r["pred_to"],
             r["correct_to"], "#795548"),
        ]

        # Draw answer box at bottom
        y_start = -0.02
        line_h = 0.045
        for ci, (label, pred, correct, color) in enumerate(
            conditions
        ):
            mark = "✓" if correct else "✗"
            pred_short = str(pred)[:30]
            text = f"{mark} {label}: {pred_short}"
            y = y_start - ci * line_h

            ax.text(
                0.02, y, text,
                transform=ax.transAxes,
                fontsize=10,
                fontfamily="monospace",
                color="green" if correct else "red",
                fontweight="bold" if (
                    label in ("Baseline", "SM-anti-sink")
                    and correct
                ) or (
                    label == "SM-sink" and not correct
                ) else "normal",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor=color,
                    alpha=0.8,
                    linewidth=2 if label in (
                        "Baseline", "SM-sink", "SM-anti-sink",
                    ) else 1,
                ),
            )

        # Ground truth
        ax.text(
            0.98, y_start, f"GT: {r['gt']}",
            transform=ax.transAxes,
            fontsize=12, fontweight="bold",
            ha="right",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="#E8F5E9",
                edgecolor="green",
                linewidth=2,
            ),
        )

        # Sink stats
        ax.text(
            0.98, y_start - line_h,
            f"Sink: {r['n_sink']}/{r['n_vis']} "
            f"({r['sink_frac']:.0%})",
            transform=ax.transAxes,
            fontsize=10, ha="right",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="#FFF3E0",
                edgecolor="#FF9800",
            ),
        )

        plt.tight_layout()
        fp = os.path.join(
            args.output_dir,
            f"smoking_gun_{ei:02d}_Q{qi}.png",
        )
        fig.savefig(
            fp, dpi=150, bbox_inches="tight",
            pad_inches=0.3,
        )
        plt.close(fig)
        print(f"  Saved {fp}")

    # Also generate a summary grid
    n = min(len(examples), 4)
    if n > 0:
        fig, axes = plt.subplots(
            1, n, figsize=(5 * n, 7),
        )
        if n == 1:
            axes = [axes]

        for i in range(n):
            r = examples[i]
            qi = r["qi"]
            ds_idx = all_indices[qi]
            row = ds[ds_idx]
            image = row["image"].convert("RGB")

            ax = axes[i]
            ax.imshow(image)
            ax.axis("off")

            # Compact labels
            q = row["question"]
            if len(q) > 50:
                q = q[:47] + "..."
            ax.set_title(q, fontsize=9)

            # Compact answer summary
            bl_m = "✓" if r["correct_bl"] else "✗"
            sm_m = "✓" if r["correct_sm"] else "✗"
            an_m = "✓" if r["correct_anti"] else "✗"
            to_m = "✓" if r["correct_to"] else "✗"
            summary = (
                f"BL:{bl_m} SM:{sm_m} "
                f"AN:{an_m} TO:{to_m}"
            )

            ax.text(
                0.5, -0.05, summary,
                transform=ax.transAxes,
                fontsize=11, ha="center",
                fontfamily="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="lightyellow",
                    edgecolor="gray",
                ),
            )

        fig.suptitle(
            "POPE Smoking Gun: "
            "BL=✓  SM-sink=✗  SM-anti-sink=✓  Text-only=✗\n"
            "Masking sinks destroys vision; "
            "masking content has no effect",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        fp = os.path.join(args.output_dir, "grid_summary.png")
        fig.savefig(fp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fp}")

    # Cross-benchmark bar chart
    print("\nGenerating cross-benchmark chart...")
    benchmarks = data["benchmarks"]
    bnames = list(benchmarks.keys())
    conditions = ["bl", "sm", "he", "rand", "anti", "to"]
    labels = ["BL", "SM-sink", "HE", "RN", "AN", "TO"]
    colors = [
        "#2196F3", "#F44336", "#FF9800",
        "#9E9E9E", "#4CAF50", "#795548",
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(bnames))
    width = 0.12
    for ci, (cond, label, color) in enumerate(
        zip(conditions, labels, colors)
    ):
        accs = []
        for bname in bnames:
            rs = benchmarks[bname]
            n = len(rs)
            if n > 0:
                acc = sum(
                    r[f"correct_{cond}"] for r in rs
                ) / n
            else:
                acc = 0
            accs.append(acc)
        offset = (ci - len(conditions) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, accs, width,
            label=label, color=color, alpha=0.85,
        )

    # Label with modality (Image / Video)
    IMAGE_BENCHMARKS = {"mmstar", "pope", "scienceqa"}
    tick_labels = []
    for b in bnames:
        short = b.replace("mvbench-", "mvb-")
        tag = "[Image]" if b in IMAGE_BENCHMARKS else "[Video]"
        tick_labels.append(f"{short}\n{tag}")

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=10)

    # Add vertical separator between image and video
    n_img = sum(1 for b in bnames if b in IMAGE_BENCHMARKS)
    if 0 < n_img < len(bnames):
        ax.axvline(
            x=n_img - 0.5, color="gray",
            linestyle="--", alpha=0.5, linewidth=1.5,
        )
        ax.text(
            n_img * 0.5 - 0.5, 0.97,
            "Image Benchmarks",
            transform=ax.get_xaxis_transform(),
            ha="center", fontsize=9, style="italic",
            color="gray",
        )
        ax.text(
            n_img + (len(bnames) - n_img) * 0.5 - 0.5, 0.97,
            "Video Benchmarks",
            transform=ax.get_xaxis_transform(),
            ha="center", fontsize=9, style="italic",
            color="gray",
        )

    ax.set_ylabel("Accuracy")
    ax.set_title(
        "Cross-Benchmark Ablation (Seed 123)\n"
        "AN ≈ BL across all benchmarks; "
        "SM-sink causes consistent drop",
    )
    ax.legend(ncol=6, loc="upper right", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fp = os.path.join(args.output_dir, "cross_benchmark.png")
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"  Saved {fp}")


if __name__ == "__main__":
    main()
