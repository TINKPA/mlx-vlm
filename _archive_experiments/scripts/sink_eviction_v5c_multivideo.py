"""
v5c: Multi-video evaluation of sink-aware KV eviction.

Tests across 6 videos (1 real + 5 synthetic) with
video-specific prompts to reduce single-video noise.

Usage:
  uv run --with "transformers<4.52" --with torch \
    --with matplotlib --with rouge-score \
    python sink_eviction_v5c_multivideo.py \
      --output-dir sink_eviction_output_v5c
"""

import argparse
import json
import os
import types
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from rouge_score import rouge_scorer

from sink_detect import (
    CaptureStore,
    find_image_token_range,
    patch_model,
    rmsnorm,
)


# ── Video-prompt pairs ──────────────────────────────

VIDEO_CONFIGS = [
    {
        "path": "test_videos/butterfly.mp4",
        "name": "butterfly",
        "prompts": [
            "Describe the butterfly in this video.",
            "What color are the butterfly's wings?",
            "What is the background scenery?",
        ],
    },
    {
        "path": "test_videos/red_car.mp4",
        "name": "red_car",
        "prompts": [
            "What is happening in this video?",
            "What colors do you see?",
            "Describe the motion in this video.",
        ],
    },
    {
        "path": "test_videos/blue_bounce.mp4",
        "name": "blue_bounce",
        "prompts": [
            "What is happening in this video?",
            "What color is the main object?",
            "Describe the background.",
        ],
    },
    {
        "path": "test_videos/color_change.mp4",
        "name": "color_change",
        "prompts": [
            "What is happening in this video?",
            "How does the color change over time?",
        ],
    },
    {
        "path": "test_videos/countdown.mp4",
        "name": "countdown",
        "prompts": [
            "What is shown in this video?",
            "What numbers or text do you see?",
        ],
    },
    {
        "path": "test_videos/split_color.mp4",
        "name": "split_color",
        "prompts": [
            "What is shown in this video?",
            "What colors are on the left and right?",
        ],
    },
]


# ── Metrics ─────────────────────────────────────────

_scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], use_stemmer=True,
)


def compute_metrics(ref, hyp):
    s = _scorer.score(ref, hyp)
    tr = set(ref.lower().split())
    th = set(hyp.lower().split())
    return {
        "rouge1_f": s["rouge1"].fmeasure,
        "rouge2_f": s["rouge2"].fmeasure,
        "rougeL_f": s["rougeL"].fmeasure,
        "jaccard": (
            len(tr & th) / len(tr | th)
            if tr and th else 0.0
        ),
    }


# ── Sink detection ──────────────────────────────────

def detect_sinks(hs, sink_dims, s, e, tau=20, layer=14):
    h = hs[layer][0]
    rms = np.abs(rmsnorm(h))
    phi = np.max(
        np.stack([rms[:, d] for d in sink_dims], -1),
        -1,
    )
    return phi[s:e]


# ── KV cache eviction ───────────────────────────────

def evict_kv_cache(cache, keep):
    keep = np.sort(keep)
    keep_mx = mx.array(keep)
    for lc in cache:
        if lc.keys is None:
            continue
        off = lc.offset
        lc.keys = lc.keys[:, :, :off, :][:, :, keep_mx, :]
        lc.values = lc.values[:, :, :off, :][:, :, keep_mx, :]
        lc.offset = len(keep)
    mx.eval([c.state for c in cache])


def gen_with_evict(
    model, processor, text,
    ids, pv, mask, evict_idx,
    max_tokens=80, **kw,
):
    from mlx_vlm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model.language_model)

    eo = model.get_input_embeddings(
        ids, pv, mask=mask, **kw,
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
    logits = out.logits[:, -1, :]

    if evict_idx is not None and len(evict_idx) > 0:
        sl = cache[0].offset
        es = set(evict_idx.tolist())
        keep = np.array(
            [i for i in range(sl) if i not in es],
            dtype=np.int32,
        )
        evict_kv_cache(cache, keep)

    y = mx.argmax(logits, axis=-1)
    tokens = [y.item()]

    for _ in range(max_tokens - 1):
        out = model.language_model(y[None], cache=cache)
        logits = out.logits[:, -1, :]
        y = mx.argmax(logits, axis=-1)
        tok = y.item()
        tokens.append(tok)
        eos = getattr(
            processor, 'eos_token_id',
            getattr(
                getattr(processor, 'tokenizer', None),
                'eos_token_id', None,
            ),
        )
        if eos is not None:
            if isinstance(eos, list):
                if tok in eos:
                    break
            elif tok == eos:
                break

    return processor.decode(tokens, skip_special_tokens=True)


# ── Strategies ──────────────────────────────────────

def strat_sink(phi, s, n, b):
    k = max(1, int(n * b))
    if k >= n:
        return None
    return np.argsort(phi)[::-1][k:] + s

def strat_rev(phi, s, n, b):
    k = max(1, int(n * b))
    if k >= n:
        return None
    return np.argsort(phi)[k:] + s

def strat_unif(phi, s, n, b):
    k = max(1, int(n * b))
    if k >= n:
        return None
    return np.arange(n - k) + s

def strat_rand(phi, s, n, b):
    k = max(1, int(n * b))
    if k >= n:
        return None
    rng = np.random.RandomState(42)
    return rng.choice(n, n - k, replace=False) + s

STRATS = {
    "sink_aware": strat_sink,
    "reverse_sink": strat_rev,
    "uniform": strat_unif,
    "random": strat_rand,
}
LABELS = {
    "sink_aware": "Sink-Aware",
    "reverse_sink": "Rev-Sink",
    "uniform": "Uniform",
    "random": "Random",
}
COLORS = {
    "sink_aware": "#2196F3",
    "reverse_sink": "#F44336",
    "uniform": "#FF9800",
    "random": "#9E9E9E",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-VL-7B-Instruct-8bit",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=80,
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        default="sink_eviction_output_v5c",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sink_dims = [458, 2570]
    detect_layer = 14
    budgets = [0.08, 0.20, 0.40, 0.60, 0.80, 1.0]

    from mlx_vlm import load
    from mlx_vlm.utils import load_config
    from mlx_vlm.video_generate import process_vision_info

    print(f"Loading {args.model}...")
    model, processor = load(args.model)
    config = load_config(args.model)
    patch_model(model)

    def prepare(video_path, prompt_text):
        msgs = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 224 * 224,
                    "fps": args.fps,
                },
                {"type": "text", "text": prompt_text},
            ],
        }]
        text = processor.apply_chat_template(
            msgs, tokenize=False,
            add_generation_prompt=True,
        )
        img_in, vid_in, _ = process_vision_info(
            msgs, True,
        )
        inputs = processor(
            text=[text], images=img_in,
            videos=vid_in, padding=True,
            return_tensors="pt",
        )
        ids = mx.array(inputs["input_ids"])
        pv_key = (
            "pixel_values_videos"
            if "pixel_values_videos" in inputs
            else "pixel_values"
        )
        pv = mx.array(inputs[pv_key])
        mask = mx.array(inputs["attention_mask"])
        extra = {}
        for k in ["video_grid_thw", "image_grid_thw"]:
            if inputs.get(k) is not None:
                extra[k] = mx.array(inputs[k])
        return text, ids, pv, mask, extra

    # ══════════════════════════════════════════════════
    # MAIN LOOP: videos x prompts x strategies x budgets
    # ══════════════════════════════════════════════════
    # Collect per-(video,prompt) ROUGE-L for each
    # (strategy, budget) pair
    all_rougeL = {
        sn: {b: [] for b in budgets}
        for sn in STRATS
    }
    all_rouge1 = {
        sn: {b: [] for b in budgets}
        for sn in STRATS
    }

    total_prompts = sum(
        len(vc["prompts"]) for vc in VIDEO_CONFIGS
    )
    total_runs = total_prompts * len(STRATS) * len(budgets)
    done = 0

    video_summaries = []

    for vc in VIDEO_CONFIGS:
        vpath = vc["path"]
        vname = vc["name"]

        if not os.path.exists(vpath):
            print(f"SKIP {vpath} (not found)")
            continue

        print(f"\n{'#'*60}")
        print(f"VIDEO: {vname} ({vpath})")
        print(f"{'#'*60}")

        # Detect sinks on first prompt
        text0, ids0, pv0, mask0, extra0 = prepare(
            vpath, vc["prompts"][0],
        )
        mcfg = model.config
        s, e = find_image_token_range(ids0, mcfg)
        nv = e - s
        print(f"  Visual: [{s},{e}) = {nv}")

        if nv == 0:
            print(f"  SKIP — no visual tokens")
            continue

        CaptureStore.enable()
        _ = gen_with_evict(
            model, processor, text0,
            ids0, pv0, mask0, None,
            max_tokens=args.max_tokens, **extra0,
        )
        CaptureStore.disable()
        hs = [np.array(h) for h in CaptureStore.hidden_states]

        if detect_layer >= len(hs):
            dl = len(hs) - 1
        else:
            dl = detect_layer

        phi = detect_sinks(
            hs, sink_dims, s, e, tau=20, layer=dl,
        )
        ns = int((phi > 20).sum())
        print(f"  Sink: {ns}/{nv} ({100*ns/nv:.0f}%)")

        v_results = []

        for prompt in vc["prompts"]:
            text_p, ids_p, pv_p, mask_p, extra_p = (
                prepare(vpath, prompt)
            )

            # Baseline
            bl = gen_with_evict(
                model, processor, text_p,
                ids_p, pv_p, mask_p, None,
                max_tokens=args.max_tokens, **extra_p,
            )
            print(f"\n  Q: {prompt}")
            print(f"  BL: {bl[:60]}")

            for sn, sfn in STRATS.items():
                for budget in budgets:
                    ev = sfn(phi, s, nv, budget)
                    if ev is None:
                        out = bl
                    else:
                        out = gen_with_evict(
                            model, processor, text_p,
                            ids_p, pv_p, mask_p, ev,
                            max_tokens=args.max_tokens,
                            **extra_p,
                        )

                    m = compute_metrics(bl, out)
                    all_rougeL[sn][budget].append(
                        m["rougeL_f"]
                    )
                    all_rouge1[sn][budget].append(
                        m["rouge1_f"]
                    )

                    done += 1
                    if done % 20 == 0 or done == total_runs:
                        print(f"    [{done}/{total_runs}]")

        video_summaries.append({
            "name": vname,
            "n_vis": nv,
            "n_sink": ns,
        })

    # ══════════════════════════════════════════════════
    # AGGREGATE
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"AGGREGATED RESULTS "
          f"({len(video_summaries)} videos, "
          f"{total_prompts} prompts)")
    print("=" * 60)

    agg = {}
    for sn in STRATS:
        agg[sn] = []
        for b in budgets:
            vals = all_rougeL[sn][b]
            agg[sn].append({
                "budget": b,
                "rougeL_mean": np.mean(vals),
                "rougeL_std": np.std(vals),
                "rouge1_mean": np.mean(
                    all_rouge1[sn][b]
                ),
                "n": len(vals),
            })

    # Print table
    print(f"\n{'Budget':<8}", end="")
    for sn in STRATS:
        print(f" {LABELS[sn]:<16}", end="")
    print()
    print("-" * 75)
    for bi, b in enumerate(budgets):
        print(f"{b:<8.0%}", end="")
        for sn in STRATS:
            m = agg[sn][bi]["rougeL_mean"]
            s = agg[sn][bi]["rougeL_std"]
            print(f" {m:.3f}±{s:.2f}    ", end="")
        print()

    # Sink-aware vs uniform delta
    print(f"\nSink-Aware vs Uniform (ROUGE-L):")
    for bi, b in enumerate(budgets):
        sa = agg["sink_aware"][bi]["rougeL_mean"]
        un = agg["uniform"][bi]["rougeL_mean"]
        d = sa - un
        sig = " ***" if abs(d) > 0.03 else ""
        print(f"  {b:>5.0%}: {d:+.3f}{sig}")

    # ══════════════════════════════════════════════════
    # PLOTS
    # ══════════════════════════════════════════════════
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Plot 1: ROUGE-L with error bars
    ax = axes[0]
    for sn in STRATS:
        bs = [a["budget"] for a in agg[sn]]
        ms = [a["rougeL_mean"] for a in agg[sn]]
        ss = [a["rougeL_std"] for a in agg[sn]]
        ax.errorbar(
            bs, ms, yerr=ss, fmt="-o",
            color=COLORS[sn], label=LABELS[sn],
            linewidth=2, markersize=5,
            capsize=3, alpha=0.85,
        )
    ax.set_xlabel("Budget (fraction kept)", fontsize=12)
    ax.set_ylabel("ROUGE-L F1", fontsize=12)
    ax.set_title(
        f"KV Cache Eviction Quality\n"
        f"({len(video_summaries)} videos, "
        f"{total_prompts} prompts)",
        fontsize=13,
    )
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    # Plot 2: Delta (sink_aware - uniform)
    ax = axes[1]
    deltas = []
    for bi, b in enumerate(budgets):
        sa = agg["sink_aware"][bi]["rougeL_mean"]
        un = agg["uniform"][bi]["rougeL_mean"]
        deltas.append(sa - un)
    colors_bar = [
        "#2196F3" if d > 0 else "#F44336"
        for d in deltas
    ]
    ax.bar(
        range(len(budgets)), deltas,
        color=colors_bar, alpha=0.7,
    )
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels(
        [f"{b:.0%}" for b in budgets], fontsize=10,
    )
    ax.set_xlabel("Budget", fontsize=12)
    ax.set_ylabel("ROUGE-L Delta", fontsize=12)
    ax.set_title(
        "Sink-Aware vs Uniform\n"
        "(blue = sink-aware wins)",
        fontsize=13,
    )
    ax.axhline(y=0, color="black", lw=0.8)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    fp = os.path.join(args.output_dir, "multivideo.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot -> {fp}")

    # Save JSON
    jp = os.path.join(args.output_dir, "results.json")
    with open(jp, "w") as f:
        json.dump({
            "videos": video_summaries,
            "budgets": budgets,
            "aggregated": {
                sn: [
                    {k: float(v) if isinstance(
                        v, (float, np.floating)
                    ) else v for k, v in a.items()}
                    for a in agg[sn]
                ]
                for sn in STRATS
            },
        }, f, indent=2)
    print(f"JSON -> {jp}")

    # Save summary
    sp = os.path.join(args.output_dir, "summary.txt")
    with open(sp, "w") as f:
        f.write("Multi-Video Sink Eviction Evaluation\n")
        f.write("=" * 60 + "\n\n")
        f.write("Videos:\n")
        for vs in video_summaries:
            f.write(f"  {vs['name']}: {vs['n_vis']} vis, "
                    f"{vs['n_sink']} sink\n")
        f.write(f"\nTotal prompts: {total_prompts}\n")
        f.write(f"Budgets: {budgets}\n\n")

        f.write("ROUGE-L (mean ± std):\n")
        f.write(f"{'Budget':<8}")
        for sn in STRATS:
            f.write(f" {LABELS[sn]:<18}")
        f.write("\n" + "-" * 80 + "\n")
        for bi, b in enumerate(budgets):
            f.write(f"{b:<8.0%}")
            for sn in STRATS:
                m = agg[sn][bi]["rougeL_mean"]
                s = agg[sn][bi]["rougeL_std"]
                f.write(f" {m:.3f}±{s:.2f}       ")
            f.write("\n")
    print(f"Summary -> {sp}")


if __name__ == "__main__":
    main()
