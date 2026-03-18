"""
v5d: Scaling test — does sink-aware advantage grow with
more visual tokens?

Tests at multiple resolutions/fps to vary token count:
  - 224², 1fps  → ~250 tokens (baseline)
  - 336², 1fps  → ~720 tokens
  - 224², 2fps  → ~500 tokens
  - 336², 2fps  → ~1440 tokens

Hypothesis: with more visual tokens, the advantage of
sink-aware eviction should be MORE pronounced because
there's more "noise" to filter.

Usage:
  uv run --with "transformers<4.52" --with torch \
    --with matplotlib --with rouge-score \
    python sink_eviction_v5d_scale.py \
      --video test_videos/butterfly.mp4 \
      --output-dir sink_eviction_output_v5d
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


_scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rougeL"], use_stemmer=True,
)


def compute_rl(ref, hyp):
    return _scorer.score(ref, hyp)["rougeL"].fmeasure


def detect_sinks(hs, dims, s, e, tau=20, layer=14):
    h = hs[min(layer, len(hs)-1)][0]
    rms = np.abs(rmsnorm(h))
    phi = np.max(
        np.stack([rms[:, d] for d in dims], -1), -1,
    )
    return phi[s:e]


def evict_kv(cache, keep):
    keep = np.sort(keep)
    kmx = mx.array(keep)
    for lc in cache:
        if lc.keys is None:
            continue
        o = lc.offset
        lc.keys = lc.keys[:, :, :o, :][:, :, kmx, :]
        lc.values = lc.values[:, :, :o, :][:, :, kmx, :]
        lc.offset = len(keep)
    mx.eval([c.state for c in cache])


def gen_evict(
    model, proc, text, ids, pv, mask,
    evict_idx, max_tok=80, **kw,
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
        evict_kv(cache, keep)

    y = mx.argmax(logits, axis=-1)
    tokens = [y.item()]
    for _ in range(max_tok - 1):
        out = model.language_model(y[None], cache=cache)
        logits = out.logits[:, -1, :]
        y = mx.argmax(logits, axis=-1)
        tok = y.item()
        tokens.append(tok)
        eos = getattr(
            proc, 'eos_token_id',
            getattr(
                getattr(proc, 'tokenizer', None),
                'eos_token_id', None,
            ),
        )
        if eos is not None:
            if isinstance(eos, list):
                if tok in eos:
                    break
            elif tok == eos:
                break
    return proc.decode(tokens, skip_special_tokens=True)


def strat_sink(phi, s, n, b):
    k = max(1, int(n * b))
    return np.argsort(phi)[::-1][k:] + s if k < n else None

def strat_unif(phi, s, n, b):
    k = max(1, int(n * b))
    return np.arange(n - k) + s if k < n else None

def strat_rand(phi, s, n, b):
    k = max(1, int(n * b))
    if k >= n:
        return None
    return np.random.RandomState(42).choice(
        n, n - k, replace=False,
    ) + s

STRATS = {
    "sink_aware": strat_sink,
    "uniform": strat_unif,
    "random": strat_rand,
}
COLORS = {
    "sink_aware": "#2196F3",
    "uniform": "#FF9800",
    "random": "#9E9E9E",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-VL-7B-Instruct-8bit",
    )
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--max-tokens", type=int, default=80,
    )
    parser.add_argument(
        "--output-dir",
        default="sink_eviction_output_v5d",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sink_dims = [458, 2570]
    budgets = [0.08, 0.20, 0.40, 0.60, 0.80, 1.0]

    # Resolution/fps configs to test
    configs = [
        {"max_pixels": 224*224, "fps": 1.0,
         "label": "224² 1fps"},
        {"max_pixels": 224*224, "fps": 2.0,
         "label": "224² 2fps"},
        {"max_pixels": 336*336, "fps": 1.0,
         "label": "336² 1fps"},
        {"max_pixels": 336*336, "fps": 2.0,
         "label": "336² 2fps"},
    ]

    prompts = [
        "Describe the butterfly in this video.",
        "What color are the butterfly's wings?",
        "What is the background scenery?",
    ]

    from mlx_vlm import load
    from mlx_vlm.utils import load_config
    from mlx_vlm.video_generate import process_vision_info

    print(f"Loading {args.model}...")
    model, processor = load(args.model)
    patch_model(model)

    def prepare(prompt, max_px, fps):
        msgs = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": args.video,
                    "max_pixels": max_px,
                    "fps": fps,
                },
                {"type": "text", "text": prompt},
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
        pvk = (
            "pixel_values_videos"
            if "pixel_values_videos" in inputs
            else "pixel_values"
        )
        pv = mx.array(inputs[pvk])
        mask = mx.array(inputs["attention_mask"])
        extra = {}
        for k in ["video_grid_thw", "image_grid_thw"]:
            if inputs.get(k) is not None:
                extra[k] = mx.array(inputs[k])
        return text, ids, pv, mask, extra

    # ══════════════════════════════════════════════════
    # MAIN: configs x prompts x strategies x budgets
    # ══════════════════════════════════════════════════
    config_results = []

    for ci, cfg in enumerate(configs):
        max_px = cfg["max_pixels"]
        fps = cfg["fps"]
        label = cfg["label"]

        print(f"\n{'#'*60}")
        print(f"CONFIG {ci+1}/{len(configs)}: {label}")
        print(f"{'#'*60}")

        # Detect sinks
        t0, ids0, pv0, m0, e0 = prepare(
            prompts[0], max_px, fps,
        )
        mcfg = model.config
        s, e = find_image_token_range(ids0, mcfg)
        nv = e - s
        print(f"  Visual tokens: {nv}")

        if nv == 0:
            print("  SKIP — no visual tokens")
            continue

        CaptureStore.enable()
        _ = gen_evict(
            model, processor, t0,
            ids0, pv0, m0, None,
            max_tok=args.max_tokens, **e0,
        )
        CaptureStore.disable()
        hs = [np.array(h) for h in CaptureStore.hidden_states]
        dl = min(14, len(hs) - 1)
        phi = detect_sinks(
            hs, sink_dims, s, e, tau=20, layer=dl,
        )
        ns = int((phi > 20).sum())
        print(f"  Sink: {ns}/{nv} ({100*ns/nv:.0f}%)")

        # Collect ROUGE-L per (strategy, budget)
        rl_data = {
            sn: {b: [] for b in budgets}
            for sn in STRATS
        }

        for pi, prompt in enumerate(prompts):
            tp, idsp, pvp, mp, ep = prepare(
                prompt, max_px, fps,
            )
            bl = gen_evict(
                model, processor, tp,
                idsp, pvp, mp, None,
                max_tok=args.max_tokens, **ep,
            )
            print(f"  P{pi+1} BL: {bl[:50]}")

            for sn, sfn in STRATS.items():
                for budget in budgets:
                    ev = sfn(phi, s, nv, budget)
                    if ev is None:
                        out = bl
                    else:
                        out = gen_evict(
                            model, processor, tp,
                            idsp, pvp, mp, ev,
                            max_tok=args.max_tokens,
                            **ep,
                        )
                    rl_data[sn][budget].append(
                        compute_rl(bl, out)
                    )

        # Aggregate for this config
        cfg_agg = {}
        for sn in STRATS:
            cfg_agg[sn] = {
                b: np.mean(rl_data[sn][b])
                for b in budgets
            }

        config_results.append({
            "label": label,
            "n_vis": nv,
            "n_sink": ns,
            "sink_pct": 100 * ns / nv,
            "agg": cfg_agg,
        })

        # Print summary for this config
        print(f"\n  {'Budget':<8} ", end="")
        for sn in STRATS:
            print(f"{sn:<14} ", end="")
        print()
        for b in budgets:
            print(f"  {b:<8.0%} ", end="")
            for sn in STRATS:
                print(f"{cfg_agg[sn][b]:<14.3f} ", end="")
            print()

    # ══════════════════════════════════════════════════
    # CROSS-CONFIG COMPARISON
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SCALING ANALYSIS: sink_aware - uniform delta")
    print("=" * 60)

    print(f"{'Config':<16} {'n_vis':>6} ", end="")
    for b in budgets:
        print(f"{b:>6.0%} ", end="")
    print()
    print("-" * 65)
    for cr in config_results:
        print(f"{cr['label']:<16} {cr['n_vis']:>6} ", end="")
        for b in budgets:
            sa = cr["agg"]["sink_aware"][b]
            un = cr["agg"]["uniform"][b]
            d = sa - un
            marker = "*" if d > 0.03 else " "
            print(f"{d:>+5.3f}{marker}", end="")
        print()

    # ══════════════════════════════════════════════════
    # PLOT
    # ══════════════════════════════════════════════════
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(config_results), figsize=(5*len(config_results), 5),
        sharey=True,
    )
    if len(config_results) == 1:
        axes = [axes]

    for ax, cr in zip(axes, config_results):
        for sn in STRATS:
            vals = [cr["agg"][sn][b] for b in budgets]
            ax.plot(
                budgets, vals, "-o",
                color=COLORS[sn], label=sn,
                linewidth=2, markersize=5,
            )
        ax.set_xlabel("Budget")
        ax.set_title(
            f"{cr['label']}\n"
            f"({cr['n_vis']} vis, "
            f"{cr['n_sink']} sink)",
            fontsize=11,
        )
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("ROUGE-L F1")
    plt.suptitle(
        "Sink-Aware Eviction: Scaling with Token Count",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()
    fp = os.path.join(args.output_dir, "scaling.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot -> {fp}")

    # Delta plot
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(budgets))
    width = 0.8 / len(config_results)
    for i, cr in enumerate(config_results):
        deltas = [
            cr["agg"]["sink_aware"][b] -
            cr["agg"]["uniform"][b]
            for b in budgets
        ]
        ax.bar(
            x + i * width - 0.4 + width/2,
            deltas, width,
            label=f"{cr['label']} ({cr['n_vis']} tok)",
            alpha=0.8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:.0%}" for b in budgets])
    ax.set_xlabel("Budget")
    ax.set_ylabel("ROUGE-L Delta (sink - uniform)")
    ax.set_title("Does Sink-Aware Advantage Scale?")
    ax.axhline(y=0, color="black", lw=0.8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fp2 = os.path.join(args.output_dir, "scaling_delta.png")
    fig.savefig(fp2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Delta plot -> {fp2}")

    # Save JSON
    jp = os.path.join(args.output_dir, "results.json")
    with open(jp, "w") as f:
        json.dump({
            "configs": [
                {
                    "label": cr["label"],
                    "n_vis": cr["n_vis"],
                    "n_sink": cr["n_sink"],
                    "agg": {
                        sn: {
                            str(b): float(v)
                            for b, v in cr["agg"][sn].items()
                        }
                        for sn in STRATS
                    },
                }
                for cr in config_results
            ],
            "budgets": budgets,
        }, f, indent=2)
    print(f"JSON -> {jp}")


if __name__ == "__main__":
    main()
