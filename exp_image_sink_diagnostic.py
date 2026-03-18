"""
Diagnostic: verify sink detection parameters on image input.

Before running multi-benchmark evaluation on images, we must
check that sink_dims and τ (calibrated on video) transfer
to images. OOD risks:
  1. Sink dims may differ (different hidden state distribution)
  2. τ=20 may be too high/low (different φ scale)
  3. Detection layer 14 may not be optimal

This script:
  1. Loads a few image samples from each benchmark
  2. Runs forward pass with hidden state capture
  3. Discovers sink dims via BOS analysis
  4. Plots φ distribution for visual tokens
  5. Compares with video sink dims [458, 2570]
  6. Recommends τ via Otsu's method if needed

Usage:
  uv run --with "transformers<4.52" --with torch \
    --with datasets --with matplotlib \
    python exp_image_sink_diagnostic.py \
      --output-dir exp_image_diagnostic
"""

import argparse
import os
import time

import mlx.core as mx
import numpy as np

from sink_detect import (
    CaptureStore,
    find_image_token_range,
    discover_sink_dims,
    rmsnorm,
)
from exp_three_way_masking import (
    patch_model_v2,
    prepare_vision_input,
    SoftMask,
)
from benchmark_loaders import (
    load_mmstar, load_pope, load_scienceqa_img,
)


def otsu_threshold(values):
    """Find optimal binary threshold via Otsu's method."""
    hist, bin_edges = np.histogram(values, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = hist.sum()
    if total == 0:
        return values.mean()
    sum_all = (hist * bin_centers).sum()
    sum_bg = 0.0
    w_bg = 0.0
    best_var = 0.0
    best_thresh = bin_centers[0]
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
        var_between = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = bin_centers[i]
    return best_thresh


def analyze_single(
    model, processor, mcfg, image, question,
    video_dims, sample_layers,
):
    """Run one image and analyze sink behavior."""
    ids, pv, mask, extra = prepare_vision_input(
        processor, question, image=image,
    )
    s, e = find_image_token_range(ids, mcfg)
    n_vis = e - s

    # Forward with capture
    CaptureStore.enable()
    SoftMask.disable()
    from mlx_vlm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model.language_model)
    eo = model.get_input_embeddings(
        ids, pv, mask=mask, **extra,
    )
    embeds = eo.inputs_embeds
    fkw = {
        k: v for k, v in eo.to_dict().items()
        if k != "inputs_embeds" and v is not None
    }
    _ = model.language_model(
        ids, inputs_embeds=embeds, cache=cache, **fkw,
    )
    mx.eval([c.state for c in cache])
    hs_np = [np.array(h) for h in CaptureStore.hidden_states]
    CaptureStore.disable()
    del cache
    mx.metal.clear_cache()

    # Discover dims from BOS
    img_dims, bos_profile = discover_sink_dims(
        hs_np, bos_idx=0, top_k=5,
    )

    # Check overlap with video dims
    overlap = set(img_dims) & set(video_dims)

    # Compute φ for visual tokens at multiple layers
    phi_by_layer = {}
    for li in sample_layers:
        if li >= len(hs_np):
            continue
        hs = hs_np[li][0]
        rms = np.abs(rmsnorm(hs))
        # Use BOTH video dims and image dims
        for dims_label, dims in [
            ("video", video_dims),
            ("image", img_dims),
        ]:
            phi = np.max(
                np.stack(
                    [rms[:, d] for d in dims], axis=-1,
                ),
                axis=-1,
            )
            vis_phi = phi[s:e]
            phi_by_layer[(li, dims_label)] = vis_phi

    return {
        "n_vis": n_vis,
        "img_dims": img_dims,
        "video_dims": video_dims,
        "dims_overlap": list(overlap),
        "phi_by_layer": phi_by_layer,
        "bos_profile": bos_profile,
        "seq_len": ids.shape[1],
        "img_start": s,
        "img_end": e,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-VL-7B-Instruct-8bit",
    )
    ap.add_argument(
        "--n-samples", type=int, default=5,
        help="Samples per benchmark for diagnostic",
    )
    ap.add_argument(
        "--output-dir", default="exp_image_diagnostic",
    )
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    video_dims = [458, 2570]
    sample_layers = [2, 5, 10, 14, 20, 25]

    # Load model
    from mlx_vlm import load
    print(f"Loading {args.model}...")
    model, processor = load(args.model)
    patch_model_v2(model)
    mcfg = model.config

    # Load a few samples from each image benchmark
    benchmarks = {
        "mmstar": load_mmstar(args.n_samples, seed=42),
        "pope": load_pope(args.n_samples, seed=42),
        "scienceqa": load_scienceqa_img(
            args.n_samples, seed=42,
        ),
    }

    all_phi_video = []  # collect all φ values (video dims)
    all_phi_image = []  # collect all φ values (image dims)
    all_discovered_dims = []

    for bname, items in benchmarks.items():
        print(f"\n{'='*50}")
        print(f"Diagnostic: {bname} ({len(items)} samples)")
        print(f"{'='*50}")

        for qi, item in enumerate(items):
            image = item["image"]
            if image is None and "_load_image" in item:
                image = item["_load_image"]()

            result = analyze_single(
                model, processor, mcfg, image,
                item["question"], video_dims,
                sample_layers,
            )

            img_dims = result["img_dims"]
            overlap = result["dims_overlap"]
            all_discovered_dims.append(img_dims)

            print(f"\n  Sample {qi}:")
            print(f"    n_vis={result['n_vis']}, "
                  f"seq_len={result['seq_len']}")
            print(f"    Image sink dims: {img_dims}")
            print(f"    Video sink dims: {video_dims}")
            print(f"    Overlap: {overlap}")

            # φ stats at layer 14
            for dims_label in ["video", "image"]:
                key = (14, dims_label)
                if key in result["phi_by_layer"]:
                    phi = result["phi_by_layer"][key]
                    n_above_20 = (phi > 20).sum()
                    n_above_10 = (phi > 10).sum()
                    print(
                        f"    φ({dims_label} dims) @ L14: "
                        f"mean={phi.mean():.1f} "
                        f"max={phi.max():.1f} "
                        f"median={np.median(phi):.1f} "
                        f">20: {n_above_20}/{len(phi)} "
                        f"({n_above_20/len(phi):.0%}) "
                        f">10: {n_above_10}/{len(phi)} "
                        f"({n_above_10/len(phi):.0%})"
                    )
                    if dims_label == "video":
                        all_phi_video.extend(phi.tolist())
                    else:
                        all_phi_image.extend(phi.tolist())

    # ── Aggregate analysis ────────────────────────────
    print(f"\n{'='*50}")
    print("AGGREGATE ANALYSIS")
    print(f"{'='*50}")

    # Dim consistency
    from collections import Counter
    dim_counts = Counter()
    for dims in all_discovered_dims:
        for d in dims:
            dim_counts[d] += 1
    print(f"\nMost frequent image sink dims:")
    for d, c in dim_counts.most_common(5):
        in_video = "✓" if d in video_dims else "✗"
        print(f"  dim {d}: found in {c}/{len(all_discovered_dims)} "
              f"samples (video: {in_video})")

    video_match = all(
        d in dim_counts for d in video_dims
    )
    print(f"\nVideo dims {video_dims} found in image data: "
          f"{'YES' if video_match else 'NO'}")

    # φ distribution (video dims)
    if all_phi_video:
        phi_v = np.array(all_phi_video)
        otsu_v = otsu_threshold(phi_v)
        n_above_20 = (phi_v > 20).sum()
        print(f"\nφ distribution (video dims, all samples):")
        print(f"  N={len(phi_v)}, "
              f"mean={phi_v.mean():.1f}, "
              f"median={np.median(phi_v):.1f}, "
              f"max={phi_v.max():.1f}")
        print(f"  >τ=20: {n_above_20}/{len(phi_v)} "
              f"({n_above_20/len(phi_v):.0%})")
        print(f"  Otsu optimal τ: {otsu_v:.1f}")

    # φ distribution (image dims)
    if all_phi_image:
        phi_i = np.array(all_phi_image)
        otsu_i = otsu_threshold(phi_i)
        n_above_20 = (phi_i > 20).sum()
        print(f"\nφ distribution (image dims, all samples):")
        print(f"  N={len(phi_i)}, "
              f"mean={phi_i.mean():.1f}, "
              f"median={np.median(phi_i):.1f}, "
              f"max={phi_i.max():.1f}")
        print(f"  >τ=20: {n_above_20}/{len(phi_i)} "
              f"({n_above_20/len(phi_i):.0%})")
        print(f"  Otsu optimal τ: {otsu_i:.1f}")

    # ── Plot ──────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if all_phi_video:
        axes[0].hist(
            all_phi_video, bins=100, alpha=0.7,
            color="steelblue",
        )
        axes[0].axvline(
            x=20, color="red", linestyle="--",
            label="τ=20 (video default)",
        )
        if all_phi_video:
            axes[0].axvline(
                x=otsu_v, color="green", linestyle="--",
                label=f"Otsu τ={otsu_v:.1f}",
            )
        axes[0].set_title("φ on IMAGE tokens (video dims)")
        axes[0].set_xlabel("φ value")
        axes[0].set_ylabel("Count")
        axes[0].legend()

    if all_phi_image:
        axes[1].hist(
            all_phi_image, bins=100, alpha=0.7,
            color="coral",
        )
        axes[1].axvline(
            x=20, color="red", linestyle="--",
            label="τ=20 (video default)",
        )
        axes[1].axvline(
            x=otsu_i, color="green", linestyle="--",
            label=f"Otsu τ={otsu_i:.1f}",
        )
        axes[1].set_title("φ on IMAGE tokens (image dims)")
        axes[1].set_xlabel("φ value")
        axes[1].legend()

    fig.suptitle(
        "OOD Diagnostic: φ distribution on image inputs\n"
        f"Video sink dims={video_dims}, "
        f"{len(all_discovered_dims)} image samples",
    )
    plt.tight_layout()
    fp = os.path.join(args.output_dir, "phi_distribution.png")
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved → {fp}")

    # ── Recommendation ────────────────────────────────
    print(f"\n{'='*50}")
    print("RECOMMENDATION")
    print(f"{'='*50}")

    if video_match:
        print("✓ Video sink dims transfer to images.")
        print("  Can use same dims [458, 2570].")
    else:
        top_dims = [d for d, _ in dim_counts.most_common(2)]
        print("✗ Video sink dims do NOT transfer.")
        print(f"  Recommend image-specific dims: {top_dims}")

    if all_phi_video:
        if abs(otsu_v - 20) < 5:
            print(f"✓ τ=20 is near Otsu optimum ({otsu_v:.1f}).")
            print("  Can use same τ=20.")
        else:
            print(f"✗ τ=20 is far from Otsu ({otsu_v:.1f}).")
            print(f"  Recommend τ={otsu_v:.0f} for images.")

    frac_20 = (
        n_above_20 / len(all_phi_video)
        if all_phi_video else 0
    )
    if 0.1 < frac_20 < 0.9:
        print(f"✓ Sink fraction at τ=20 is {frac_20:.0%} "
              f"(reasonable).")
    elif frac_20 >= 0.9:
        print(f"⚠ {frac_20:.0%} classified as sink at τ=20"
              f" — too many. Raise τ.")
    else:
        print(f"⚠ Only {frac_20:.0%} classified as sink "
              f"at τ=20 — too few. Lower τ.")


if __name__ == "__main__":
    main()
