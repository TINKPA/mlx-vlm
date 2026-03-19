"""
Detailed smoking gun visualization with spatial sink maps.

For each example, generates a 2x2 panel:
  (a) Original + sink overlay (red=sink, green=content)
  (b) Baseline view → answer
  (c) SM-sink "view": content bright, sink dimmed → answer
  (d) SM-anti-sink "view": sink bright, content dimmed → answer

Requires running forward pass to get sink spatial positions.

Usage:
  uv run --with "transformers<4.52" --with torch \
    --with datasets --with matplotlib --with pillow \
    python viz_detailed_smoking_gun.py \
      --results-json .../results.json \
      --output-dir viz_detailed \
      --n-examples 5
"""

import argparse
import json
import os

import mlx.core as mx
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-json", required=True)
    ap.add_argument("--output-dir", default="viz_detailed")
    ap.add_argument("--n-examples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-VL-7B-Instruct-8bit",
    )
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    # Load results — find smoking guns
    data = json.load(open(args.results_json))
    pope = data["benchmarks"]["pope"]
    smoking = [
        r for r in pope
        if r["correct_bl"]
        and not r["correct_sm"]
        and r["correct_anti"]
    ]
    print(f"Found {len(smoking)} smoking gun examples")
    examples = smoking[:args.n_examples]

    # Load POPE dataset
    from datasets import load_dataset
    print("Loading POPE dataset...")
    ds = load_dataset("lmms-lab/POPE", split="test")
    rng = np.random.RandomState(args.seed)
    all_indices = rng.choice(
        len(ds), min(100, len(ds)), replace=False,
    ).tolist()

    # Load model for sink detection
    from mlx_vlm import load
    from sink_detect import (
        CaptureStore, find_image_token_range, rmsnorm,
    )
    from exp_three_way_masking import (
        patch_model_v2, prepare_vision_input, SoftMask,
    )

    print(f"Loading {args.model}...")
    model, processor = load(args.model)
    patch_model_v2(model)
    mcfg = model.config
    sink_dims = [458, 2570]
    sms = mcfg.vision_config.spatial_merge_size

    for ei, r in enumerate(examples):
        qi = r["qi"]
        ds_idx = all_indices[qi]
        row = ds[ds_idx]
        pil_image = row["image"].convert("RGB")
        question = row["question"]
        gt = r["gt"]

        print(f"\n--- Example {ei}: Q{qi} ---")
        print(f"  Question: {question}")
        print(f"  GT: {gt}")

        # Run forward pass to get sink positions
        ids, pv, attn_mask, extra = prepare_vision_input(
            processor, question, image=pil_image,
        )
        s, e = find_image_token_range(ids, mcfg)
        n_vis = e - s

        CaptureStore.enable()
        SoftMask.disable()
        from mlx_vlm.models.cache import make_prompt_cache
        cache = make_prompt_cache(model.language_model)
        eo = model.get_input_embeddings(
            ids, pv, mask=attn_mask, **extra,
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
        hs_np = [
            np.array(h) for h in CaptureStore.hidden_states
        ]
        CaptureStore.disable()
        del cache
        mx.clear_cache()

        # Detect sinks at layer 14
        hs = hs_np[14][0]
        rms_val = np.abs(rmsnorm(hs))
        phi = np.max(
            np.stack(
                [rms_val[:, d] for d in sink_dims], axis=-1,
            ),
            axis=-1,
        )
        vis_phi = phi[s:e]
        sink_mask = vis_phi > 20.0
        n_sink = sink_mask.sum()
        print(f"  Sinks: {n_sink}/{n_vis} "
              f"({n_sink/n_vis:.0%})")

        # Get grid dimensions for spatial mapping
        thw_key = (
            "image_grid_thw"
            if "image_grid_thw" in extra
            else "video_grid_thw"
        )
        grid_thw = extra[thw_key]
        T = int(grid_thw[0, 0])
        H = int(grid_thw[0, 1])
        W = int(grid_thw[0, 2])
        llm_H = H // sms
        llm_W = W // sms
        tokens_per_frame = llm_H * llm_W

        # Reshape sink mask to spatial grid
        if n_vis >= tokens_per_frame:
            # Use first frame
            frame_sink = sink_mask[:tokens_per_frame]
            frame_phi = vis_phi[:tokens_per_frame]
            sink_grid = frame_sink.reshape(llm_H, llm_W)
            phi_grid = frame_phi.reshape(llm_H, llm_W)
        else:
            # Fallback: reshape to closest rectangle
            h = int(np.sqrt(n_vis))
            w = n_vis // h
            sink_grid = sink_mask[:h * w].reshape(h, w)
            phi_grid = vis_phi[:h * w].reshape(h, w)

        del hs_np

        # Create visualization
        img_w, img_h = pil_image.size
        img_arr = np.array(pil_image).astype(np.float32)

        # Upsample sink grid to image size
        import cv2
        sink_up = cv2.resize(
            sink_grid.astype(np.float32),
            (img_w, img_h),
            interpolation=cv2.INTER_NEAREST,
        )
        phi_up = cv2.resize(
            phi_grid.astype(np.float32),
            (img_w, img_h),
            interpolation=cv2.INTER_NEAREST,
        )

        # Create overlays
        # (a) Sink map: red=sink, green=content
        overlay_a = img_arr.copy()
        # Red tint on sinks
        overlay_a[sink_up > 0.5, 0] = np.clip(
            overlay_a[sink_up > 0.5, 0] * 0.5 + 128, 0, 255,
        )
        overlay_a[sink_up > 0.5, 1] *= 0.4
        overlay_a[sink_up > 0.5, 2] *= 0.4
        # Green tint on content
        overlay_a[sink_up < 0.5, 1] = np.clip(
            overlay_a[sink_up < 0.5, 1] * 0.5 + 128, 0, 255,
        )
        overlay_a[sink_up < 0.5, 0] *= 0.5
        overlay_a[sink_up < 0.5, 2] *= 0.5

        # (c) SM-sink view: content bright, sinks very dark
        view_sm = img_arr.copy()
        view_sm[sink_up > 0.5] *= 0.15  # dim sinks heavily
        # Add red border on dimmed area
        # Find edges of sink region
        from scipy import ndimage
        edges = ndimage.binary_dilation(
            sink_up > 0.5, iterations=2,
        ) ^ (sink_up > 0.5)
        view_sm[edges, 0] = 255
        view_sm[edges, 1] = 50
        view_sm[edges, 2] = 50

        # (d) SM-anti-sink view: sinks bright, content dark
        view_an = img_arr.copy()
        view_an[sink_up < 0.5] *= 0.15  # dim content
        edges_c = ndimage.binary_dilation(
            sink_up < 0.5, iterations=2,
        ) ^ (sink_up < 0.5)
        view_an[edges_c, 0] = 50
        view_an[edges_c, 1] = 50
        view_an[edges_c, 2] = 255

        # ── Draw 2×2 figure ───────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # (a) Original + sink map
        axes[0, 0].imshow(overlay_a.astype(np.uint8))
        axes[0, 0].set_title(
            f"(a) Sink Map: "
            f"red = sink ({n_sink}/{n_vis}, "
            f"{n_sink/n_vis:.0%}), "
            f"green = content",
            fontsize=11,
        )
        axes[0, 0].axis("off")

        # (b) Baseline — original image, correct
        axes[0, 1].imshow(pil_image)
        axes[0, 1].set_title(
            f"(b) Baseline → \"{r['pred_bl']}\" ✓",
            fontsize=12, color="green", fontweight="bold",
        )
        axes[0, 1].axis("off")

        # (c) SM-sink view — content visible, answer wrong
        axes[1, 0].imshow(view_sm.astype(np.uint8))
        axes[1, 0].set_title(
            f"(c) SM-sink (mask sinks, keep content) "
            f"→ \"{r['pred_sm']}\" ✗",
            fontsize=12, color="red", fontweight="bold",
        )
        axes[1, 0].axis("off")
        axes[1, 0].text(
            0.5, 0.02,
            "Content patches visible, but model can't "
            "answer without sinks",
            transform=axes[1, 0].transAxes,
            fontsize=9, ha="center", style="italic",
            color="white",
            bbox=dict(facecolor="red", alpha=0.7),
        )

        # (d) SM-anti-sink view — sinks visible, correct
        axes[1, 1].imshow(view_an.astype(np.uint8))
        axes[1, 1].set_title(
            f"(d) SM-anti-sink (mask content, keep sinks) "
            f"→ \"{r['pred_anti']}\" ✓",
            fontsize=12, color="green", fontweight="bold",
        )
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.5, 0.02,
            "Most patches masked, but sinks alone "
            "are enough to answer correctly",
            transform=axes[1, 1].transAxes,
            fontsize=9, ha="center", style="italic",
            color="white",
            bbox=dict(facecolor="green", alpha=0.7),
        )

        # Suptitle
        q_disp = question
        if len(q_disp) > 80:
            q_disp = q_disp[:77] + "..."
        fig.suptitle(
            f"Q: \"{q_disp}\"    GT: {gt}    "
            f"Text-only: \"{r['pred_to'][:30]}\" ✗\n"
            f"Sink fraction: {n_sink/n_vis:.0%} — "
            f"Visual info flows through sinks, "
            f"not content tokens",
            fontsize=13, fontweight="bold",
        )

        plt.tight_layout()
        fp = os.path.join(
            args.output_dir,
            f"detailed_{ei:02d}_Q{qi}.png",
        )
        fig.savefig(
            fp, dpi=150, bbox_inches="tight",
        )
        plt.close(fig)
        print(f"  Saved {fp}")

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
