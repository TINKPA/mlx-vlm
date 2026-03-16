"""
POC v3: Deeper sink eviction analysis.

1. Fine-grained questions requiring visual details
2. Graduated eviction (keep top-K sinks by phi)
3. Multiple detection layers
4. Quantitative: token overlap with baseline

Usage:
  uv run --with "transformers<4.52" --with torch \
    --with matplotlib \
    python sink_eviction_poc_v3.py \
      --video test_video.mp4 \
      --output-dir sink_eviction_output_v3
"""

import argparse
import os
import types
from typing import List, Tuple

import mlx.core as mx
import numpy as np

from sink_detect import (
    CaptureStore,
    find_image_token_range,
    patch_model,
    rmsnorm,
)


# ── Embedding zeroing patch ──────────────────────────

_original_get_input_embeddings = None
_zero_out_indices = None


def _patched_get_input_embeddings(
    self, input_ids=None, pixel_values=None, **kwargs
):
    result = _original_get_input_embeddings(
        self, input_ids, pixel_values, **kwargs,
    )
    if _zero_out_indices is not None and len(
        _zero_out_indices
    ) > 0:
        embeds = result.inputs_embeds
        seq_len = embeds.shape[1]
        keep = np.ones((1, seq_len, 1), dtype=np.float32)
        for idx in _zero_out_indices:
            if idx < seq_len:
                keep[0, idx, 0] = 0.0
        result.inputs_embeds = embeds * mx.array(keep)
    return result


def run_gen(
    model, processor, text, gen_kwargs,
    drop_indices=None,
):
    global _zero_out_indices
    from mlx_vlm import generate
    _zero_out_indices = (
        drop_indices if drop_indices is not None
        else np.array([])
    )
    output = generate(
        model, processor, prompt=text,
        verbose=False, **gen_kwargs,
    )
    _zero_out_indices = None
    return output.text


def get_phi_and_ranks(
    hidden_states, sink_dims, img_start, img_end, layer,
):
    """Get phi values and ranked indices for visual tokens."""
    hs = hidden_states[layer][0]
    rms = np.abs(rmsnorm(hs))
    phi = np.max(
        np.stack(
            [rms[:, d] for d in sink_dims], axis=-1
        ),
        axis=-1,
    )
    vis_phi = phi[img_start:img_end]
    # Rank by phi descending (highest phi = strongest sink)
    ranked_local = np.argsort(vis_phi)[::-1]
    return vis_phi, ranked_local


def token_overlap(a: str, b: str) -> float:
    """Jaccard similarity of token sets."""
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


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
        "--max-tokens", type=int, default=80,
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument(
        "--output-dir", default="sink_eviction_output_v3",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sink_dims = [458, 2570]

    # Multiple prompts: generic → fine-grained
    prompts = [
        "Describe the butterfly in this video.",
        "What color are the butterfly's wings?",
        "What is the background scenery in this video?",
        "How many characters are in this video?",
        "Is the butterfly on the left or right side?",
    ]

    # ── Load model ──
    from mlx_vlm import load, generate
    from mlx_vlm.utils import load_config
    from mlx_vlm.video_generate import process_vision_info

    print(f"Loading {args.model}...")
    model, processor = load(args.model)
    config = load_config(args.model)
    patch_model(model)

    global _original_get_input_embeddings
    _original_get_input_embeddings = (
        model.get_input_embeddings.__func__
    )

    def _bound_patch(self, input_ids=None,
                     pixel_values=None, **kwargs):
        return _patched_get_input_embeddings(
            self, input_ids, pixel_values, **kwargs,
        )

    model.get_input_embeddings = types.MethodType(
        _bound_patch, model,
    )

    # ── Prepare video input ──
    def prepare_inputs(prompt_text):
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
                    {"type": "text", "text": prompt_text},
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

        extra = {}
        for key in ["video_grid_thw", "image_grid_thw"]:
            if inputs.get(key) is not None:
                extra[key] = mx.array(inputs[key])

        gen_kwargs = dict(extra)
        gen_kwargs["video"] = [args.video]
        gen_kwargs["input_ids"] = input_ids
        gen_kwargs["pixel_values"] = pixel_values
        gen_kwargs["mask"] = mask_input
        gen_kwargs["temperature"] = 0.0
        gen_kwargs["max_tokens"] = args.max_tokens

        return text, input_ids, gen_kwargs

    # ── First pass: get hidden states for sink detection ──
    print("First pass for sink detection...")
    text0, input_ids0, gen_kwargs0 = prepare_inputs(
        prompts[0]
    )
    model_config = model.config
    img_start, img_end = find_image_token_range(
        input_ids0, model_config,
    )
    n_vis = img_end - img_start
    print(f"Visual tokens: [{img_start},{img_end}) "
          f"= {n_vis}")

    CaptureStore.enable()
    baseline0 = run_gen(
        model, processor, text0, gen_kwargs0,
    )
    CaptureStore.disable()

    hs_np = [
        np.array(h) for h in CaptureStore.hidden_states
    ]

    # ══════════════════════════════════════════════════
    # EXPERIMENT 1: Graduated eviction at Layer 14
    # Keep top-K sink tokens, zero the rest of visual
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("EXP 1: Graduated eviction — how many sinks "
          "do we need?")
    print("=" * 60)

    detect_layer = 14
    vis_phi, ranked_local = get_phi_and_ranks(
        hs_np, sink_dims, img_start, img_end,
        detect_layer,
    )

    # Test keeping different numbers of top-phi tokens
    keep_counts = [0, 5, 10, 20, 50, 100, 150, 200, 250]
    keep_counts = [k for k in keep_counts if k <= n_vis]

    print(f"\nPrompt: {prompts[0]}")
    print(f"Baseline: {baseline0[:70]}...")
    print()

    grad_results = []
    for k in keep_counts:
        if k == n_vis:
            # Keep all = baseline
            out = baseline0
        elif k == 0:
            # Drop all
            all_vis = np.arange(img_start, img_end)
            out = run_gen(
                model, processor, text0, gen_kwargs0,
                drop_indices=all_vis,
            )
        else:
            # Keep top-K by phi, zero the rest
            keep_local = ranked_local[:k]
            drop_local = ranked_local[k:]
            drop_abs = drop_local + img_start
            out = run_gen(
                model, processor, text0, gen_kwargs0,
                drop_indices=drop_abs,
            )

        sim = token_overlap(baseline0, out)
        coherent = (
            "..." not in out[:30]
            and len(set(out.split())) > 5
        )
        grad_results.append((k, out, sim, coherent))

        status = "OK" if coherent else "BROKEN"
        print(f"  Keep {k:>3}/{n_vis}: "
              f"sim={sim:.2f} [{status}] "
              f"{out[:50]}")

    # ══════════════════════════════════════════════════
    # EXPERIMENT 2: Multi-prompt with sink eviction
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("EXP 2: Multiple prompts — sink vs non-sink "
          "eviction")
    print("=" * 60)

    # Use Layer 14 detection, keep only sink tokens
    tau = 20.0
    sink_mask = vis_phi > tau
    sink_local = np.where(sink_mask)[0]
    nonsink_local = np.where(~sink_mask)[0]
    sink_abs = sink_local + img_start
    nonsink_abs = nonsink_local + img_start
    n_sink = len(sink_abs)
    n_nonsink = len(nonsink_abs)

    print(f"Sink: {n_sink}, Non-sink: {n_nonsink}")

    prompt_results = []
    for prompt in prompts:
        text_p, _, gen_kwargs_p = prepare_inputs(prompt)

        # Baseline
        out_base = run_gen(
            model, processor, text_p, gen_kwargs_p,
        )
        # Drop sink
        out_drop_sink = run_gen(
            model, processor, text_p, gen_kwargs_p,
            drop_indices=sink_abs,
        )
        # Drop non-sink
        out_drop_nonsink = run_gen(
            model, processor, text_p, gen_kwargs_p,
            drop_indices=nonsink_abs,
        )

        sim_sink = token_overlap(out_base, out_drop_sink)
        sim_nonsink = token_overlap(
            out_base, out_drop_nonsink
        )

        prompt_results.append({
            "prompt": prompt,
            "baseline": out_base,
            "drop_sink": out_drop_sink,
            "drop_nonsink": out_drop_nonsink,
            "sim_drop_sink": sim_sink,
            "sim_drop_nonsink": sim_nonsink,
        })

        print(f"\n  Q: {prompt}")
        print(f"  Baseline:     {out_base[:60]}")
        print(f"  Drop sink:    {out_drop_sink[:60]}")
        print(f"  Drop nonsink: {out_drop_nonsink[:60]}")
        print(f"  Similarity — drop_sink={sim_sink:.2f}, "
              f"drop_nonsink={sim_nonsink:.2f}")

    # ══════════════════════════════════════════════════
    # EXPERIMENT 3: Detection layer comparison
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("EXP 3: Which detection layer is best?")
    print("=" * 60)

    test_layers = [5, 10, 14, 20, 25]
    test_layers = [
        l for l in test_layers if l < len(hs_np)
    ]

    text_p, _, gen_kwargs_p = prepare_inputs(prompts[0])
    layer_results = []

    for li in test_layers:
        vp, rl = get_phi_and_ranks(
            hs_np, sink_dims, img_start, img_end, li,
        )
        sink_m = vp > tau
        sink_l = np.where(sink_m)[0]
        nonsink_l = np.where(~sink_m)[0]
        n_s = len(sink_l)
        n_ns = len(nonsink_l)

        # Drop sink
        out_ds = run_gen(
            model, processor, text_p, gen_kwargs_p,
            drop_indices=sink_l + img_start,
        )
        sim_ds = token_overlap(baseline0, out_ds)
        coherent_ds = len(set(out_ds.split())) > 5

        # Drop non-sink
        out_dns = run_gen(
            model, processor, text_p, gen_kwargs_p,
            drop_indices=nonsink_l + img_start,
        )
        sim_dns = token_overlap(baseline0, out_dns)

        layer_results.append({
            "layer": li,
            "n_sink": n_s,
            "n_nonsink": n_ns,
            "sim_drop_sink": sim_ds,
            "sim_drop_nonsink": sim_dns,
            "drop_sink_coherent": coherent_ds,
        })

        print(f"  L{li:>2}: sink={n_s:>3} "
              f"nonsink={n_ns:>3} | "
              f"drop_sink_sim={sim_ds:.2f} "
              f"({'OK' if coherent_ds else 'BROKEN'}) | "
              f"drop_nonsink_sim={sim_dns:.2f}")

    # ── Summary plot ──
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Graduated eviction
    ax = axes[0]
    ks = [r[0] for r in grad_results]
    sims = [r[2] for r in grad_results]
    colors = [
        "green" if r[3] else "red"
        for r in grad_results
    ]
    ax.bar(range(len(ks)), sims, color=colors, alpha=0.7)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(
        [str(k) for k in ks], fontsize=8,
    )
    ax.set_xlabel("# Visual Tokens Kept (by φ rank)")
    ax.set_ylabel("Token Overlap with Baseline")
    ax.set_title(
        "How Many Sink Tokens Are Enough?\n"
        "Green=coherent, Red=broken",
    )
    ax.set_ylim(0, 1.05)

    # Plot 2: Multi-prompt comparison
    ax = axes[1]
    x_pos = np.arange(len(prompts))
    w = 0.35
    sim_s = [r["sim_drop_sink"] for r in prompt_results]
    sim_ns = [
        r["sim_drop_nonsink"] for r in prompt_results
    ]
    ax.bar(
        x_pos - w / 2, sim_s, w,
        label="Drop Sink", color="red", alpha=0.7,
    )
    ax.bar(
        x_pos + w / 2, sim_ns, w,
        label="Drop Non-Sink", color="green", alpha=0.7,
    )
    ax.set_xticks(x_pos)
    short = [p[:20] + "..." for p in prompts]
    ax.set_xticklabels(short, fontsize=7, rotation=30,
                       ha="right")
    ax.set_ylabel("Token Overlap with Baseline")
    ax.set_title("Drop Sink vs Drop Non-Sink\n"
                 "Across Different Prompts")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    # Plot 3: Detection layer
    ax = axes[2]
    ls = [r["layer"] for r in layer_results]
    sim_ds = [r["sim_drop_sink"] for r in layer_results]
    sim_dns = [
        r["sim_drop_nonsink"] for r in layer_results
    ]
    ax.plot(ls, sim_ds, "r-o", label="Drop Sink")
    ax.plot(ls, sim_dns, "g-o", label="Drop Non-Sink")
    ax.set_xlabel("Detection Layer")
    ax.set_ylabel("Token Overlap with Baseline")
    ax.set_title("Effect of Detection Layer\n"
                 "on Eviction Quality")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(args.output_dir, "summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved → {path}")

    # ── Save text results ──
    path = os.path.join(args.output_dir, "results.txt")
    with open(path, "w") as f:
        f.write("Sink Eviction POC v3 — Deep Analysis\n")
        f.write("=" * 60 + "\n\n")

        f.write("EXP 1: Graduated Eviction\n")
        f.write("-" * 40 + "\n")
        for k, out, sim, coh in grad_results:
            f.write(f"Keep {k:>3}: sim={sim:.2f} "
                    f"{'OK' if coh else 'BROKEN'}\n")
            f.write(f"  {out[:80]}\n\n")

        f.write("\nEXP 2: Multi-Prompt\n")
        f.write("-" * 40 + "\n")
        for r in prompt_results:
            f.write(f"\nQ: {r['prompt']}\n")
            f.write(f"Baseline: {r['baseline'][:80]}\n")
            f.write(f"Drop sink (sim={r['sim_drop_sink']:.2f}): "
                    f"{r['drop_sink'][:80]}\n")
            f.write(f"Drop non-sink "
                    f"(sim={r['sim_drop_nonsink']:.2f}): "
                    f"{r['drop_nonsink'][:80]}\n")

        f.write("\n\nEXP 3: Detection Layer\n")
        f.write("-" * 40 + "\n")
        for r in layer_results:
            f.write(f"L{r['layer']:>2}: "
                    f"sink={r['n_sink']:>3} "
                    f"drop_sink_sim={r['sim_drop_sink']:.2f} "
                    f"drop_nonsink_sim="
                    f"{r['sim_drop_nonsink']:.2f}\n")

    print(f"Saved → {path}")


if __name__ == "__main__":
    main()
