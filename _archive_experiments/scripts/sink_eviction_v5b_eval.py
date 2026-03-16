"""
v5b: Systematic evaluation of sink-aware KV cache eviction.

Improvements over v5:
  - ROUGE-L metric (captures sequence-level similarity)
  - Multiple prompts (generic, color, scene, spatial, count)
  - Finer budget grid around the sweet spot
  - Aggregated scores across prompts
  - Publication-quality figures

Usage:
  uv run --with "transformers<4.52" --with torch \
    --with matplotlib --with rouge-score \
    python sink_eviction_v5b_eval.py \
      --video test_video.mp4 \
      --output-dir sink_eviction_output_v5b
"""

import argparse
import json
import os
import types
from typing import List, Optional, Tuple

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


# ── Metrics ─────────────────────────────────────────

_scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], use_stemmer=True,
)


def compute_metrics(reference: str, hypothesis: str):
    """Compute ROUGE + Jaccard metrics."""
    scores = _scorer.score(reference, hypothesis)
    tr = set(reference.lower().split())
    th = set(hypothesis.lower().split())
    jaccard = (
        len(tr & th) / len(tr | th)
        if tr and th else 0.0
    )
    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rouge2_f": scores["rouge2"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure,
        "jaccard": jaccard,
    }


# ── Sink detection ──────────────────────────────────

def detect_sinks(
    hidden_states, sink_dims, img_start, img_end,
    tau=20.0, layer=14,
):
    hs = hidden_states[layer][0]
    rms = np.abs(rmsnorm(hs))
    phi = np.max(
        np.stack(
            [rms[:, d] for d in sink_dims], axis=-1
        ),
        axis=-1,
    )
    vis_phi = phi[img_start:img_end]
    sink_mask = vis_phi > tau
    return vis_phi, sink_mask


# ── KV cache eviction ───────────────────────────────

def evict_kv_cache(prompt_cache, keep_indices):
    keep = np.sort(keep_indices)
    keep_mx = mx.array(keep)

    for lc in prompt_cache:
        if lc.keys is None:
            continue
        offset = lc.offset
        k = lc.keys[:, :, :offset, :]
        v = lc.values[:, :, :offset, :]
        lc.keys = k[:, :, keep_mx, :]
        lc.values = v[:, :, keep_mx, :]
        lc.offset = len(keep)

    mx.eval([c.state for c in prompt_cache])


def generate_with_eviction(
    model, processor, prompt_text,
    input_ids, pixel_values, mask,
    evict_indices, max_tokens=100,
    temperature=0.0, **extra_kwargs,
):
    from mlx_vlm.models.cache import make_prompt_cache

    prompt_cache = make_prompt_cache(
        model.language_model,
    )

    embedding_output = model.get_input_embeddings(
        input_ids, pixel_values,
        mask=mask, **extra_kwargs,
    )
    inputs_embeds = embedding_output.inputs_embeds
    fwd_kwargs = {
        k: v
        for k, v in embedding_output.to_dict().items()
        if k != "inputs_embeds" and v is not None
    }

    outputs = model.language_model(
        input_ids, inputs_embeds=inputs_embeds,
        cache=prompt_cache, **fwd_kwargs,
    )
    mx.eval([c.state for c in prompt_cache])
    logits = outputs.logits[:, -1, :]

    if evict_indices is not None and len(
        evict_indices
    ) > 0:
        seq_len = prompt_cache[0].offset
        evict_set = set(evict_indices.tolist())
        keep = np.array(
            [i for i in range(seq_len)
             if i not in evict_set],
            dtype=np.int32,
        )
        evict_kv_cache(prompt_cache, keep)

    if temperature == 0:
        y = mx.argmax(logits, axis=-1)
    else:
        y = mx.random.categorical(logits / temperature)

    tokens = [y.item()]

    for _ in range(max_tokens - 1):
        outputs = model.language_model(
            y[None], cache=prompt_cache,
        )
        logits = outputs.logits[:, -1, :]
        if temperature == 0:
            y = mx.argmax(logits, axis=-1)
        else:
            y = mx.random.categorical(
                logits / temperature
            )
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

    return processor.decode(
        tokens, skip_special_tokens=True,
    )


# ── Strategies ──────────────────────────────────────

def strat_sink_aware(phi, img_start, n_vis, budget):
    n_keep = max(1, int(n_vis * budget))
    if n_keep >= n_vis:
        return None
    ranked = np.argsort(phi)[::-1]
    return ranked[n_keep:] + img_start


def strat_reverse_sink(phi, img_start, n_vis, budget):
    n_keep = max(1, int(n_vis * budget))
    if n_keep >= n_vis:
        return None
    ranked = np.argsort(phi)  # ascending
    return ranked[n_keep:] + img_start


def strat_uniform(phi, img_start, n_vis, budget):
    n_keep = max(1, int(n_vis * budget))
    if n_keep >= n_vis:
        return None
    n_evict = n_vis - n_keep
    return np.arange(n_evict) + img_start


def strat_random(phi, img_start, n_vis, budget):
    n_keep = max(1, int(n_vis * budget))
    if n_keep >= n_vis:
        return None
    rng = np.random.RandomState(42)
    return rng.choice(
        n_vis, size=n_vis - n_keep, replace=False,
    ) + img_start


STRATEGIES = {
    "sink_aware": strat_sink_aware,
    "reverse_sink": strat_reverse_sink,
    "uniform": strat_uniform,
    "random": strat_random,
}

LABELS = {
    "sink_aware": "Sink-Aware (ours)",
    "reverse_sink": "Reverse-Sink",
    "uniform": "Uniform Oldest",
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
        default=(
            "mlx-community/Qwen2.5-VL-7B-Instruct-8bit"
        ),
    )
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--max-tokens", type=int, default=100,
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        default="sink_eviction_output_v5b",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sink_dims = [458, 2570]
    detect_layer = 14

    # Finer budget grid with more points around sweet spot
    budgets = [
        0.04, 0.08, 0.12, 0.20, 0.30,
        0.40, 0.50, 0.60, 0.80, 1.0,
    ]

    prompts = [
        "Describe the butterfly in this video.",
        "What color are the butterfly's wings?",
        "What is the background scenery?",
        "Describe all characters in this video.",
        "Is the butterfly on the left or right side "
        "of the screen?",
    ]

    # ── Load model ──
    from mlx_vlm import load
    from mlx_vlm.utils import load_config
    from mlx_vlm.video_generate import process_vision_info

    print(f"Loading {args.model}...")
    model, processor = load(args.model)
    config = load_config(args.model)
    patch_model(model)

    # ── Helper to prepare inputs per prompt ──
    def prepare(prompt_text):
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

        return text, input_ids, pixel_values, mask_input, extra

    # ── Sink detection using first prompt ──
    print("Detecting sinks...")
    text0, ids0, pv0, mask0, extra0 = prepare(prompts[0])
    model_config = model.config
    img_start, img_end = find_image_token_range(
        ids0, model_config,
    )
    n_vis = img_end - img_start
    print(f"Visual: [{img_start},{img_end}) = {n_vis}")

    CaptureStore.enable()
    baseline0 = generate_with_eviction(
        model, processor, text0,
        ids0, pv0, mask0,
        evict_indices=None,
        max_tokens=args.max_tokens, **extra0,
    )
    CaptureStore.disable()

    hs_np = [
        np.array(h) for h in CaptureStore.hidden_states
    ]
    vis_phi, sink_mask = detect_sinks(
        hs_np, sink_dims, img_start, img_end,
        tau=20.0, layer=detect_layer,
    )
    n_sink = int(sink_mask.sum())
    print(f"Sink: {n_sink}/{n_vis} "
          f"({100*n_sink/n_vis:.0f}%)")

    # ══════════════════════════════════════════════════
    # MAIN LOOP: prompts x strategies x budgets
    # ══════════════════════════════════════════════════
    # Structure: all_results[prompt_idx][strat][budget_idx]
    all_results = []
    baselines = []

    total = len(prompts) * len(STRATEGIES) * len(budgets)
    done = 0

    for pi, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"Prompt {pi+1}/{len(prompts)}: {prompt}")
        print(f"{'='*60}")

        text_p, ids_p, pv_p, mask_p, extra_p = (
            prepare(prompt)
        )

        # Baseline for this prompt
        bl = generate_with_eviction(
            model, processor, text_p,
            ids_p, pv_p, mask_p,
            evict_indices=None,
            max_tokens=args.max_tokens, **extra_p,
        )
        baselines.append(bl)
        print(f"  Baseline: {bl[:60]}")

        prompt_results = {}
        for sn, sfn in STRATEGIES.items():
            prompt_results[sn] = []
            for budget in budgets:
                evict = sfn(
                    vis_phi, img_start, n_vis, budget,
                )
                if evict is None:
                    out = bl
                else:
                    out = generate_with_eviction(
                        model, processor, text_p,
                        ids_p, pv_p, mask_p,
                        evict_indices=evict,
                        max_tokens=args.max_tokens,
                        **extra_p,
                    )

                m = compute_metrics(bl, out)
                n_keep = max(1, int(n_vis * budget))
                prompt_results[sn].append({
                    "budget": budget,
                    "n_keep": n_keep,
                    "output": out,
                    **m,
                })

                done += 1
                print(
                    f"  [{done}/{total}] {sn:<14} "
                    f"{budget:.0%} "
                    f"R1={m['rouge1_f']:.2f} "
                    f"RL={m['rougeL_f']:.2f} "
                    f"{out[:40]}"
                )

        all_results.append(prompt_results)

    # ══════════════════════════════════════════════════
    # AGGREGATE SCORES
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS (mean across prompts)")
    print("=" * 60)

    agg = {}  # agg[strat][budget_idx] = mean metrics
    for sn in STRATEGIES:
        agg[sn] = []
        for bi in range(len(budgets)):
            r1s, r2s, rls, jacs = [], [], [], []
            for pi in range(len(prompts)):
                r = all_results[pi][sn][bi]
                r1s.append(r["rouge1_f"])
                r2s.append(r["rouge2_f"])
                rls.append(r["rougeL_f"])
                jacs.append(r["jaccard"])
            agg[sn].append({
                "budget": budgets[bi],
                "rouge1_f": np.mean(r1s),
                "rouge2_f": np.mean(r2s),
                "rougeL_f": np.mean(rls),
                "jaccard": np.mean(jacs),
            })

    # Print aggregated table
    header = f"{'Budget':<8}"
    for sn in STRATEGIES:
        header += f" {LABELS[sn]:<20}"
    print(header)
    print("-" * len(header))

    for bi in range(len(budgets)):
        row = f"{budgets[bi]:<8.0%}"
        for sn in STRATEGIES:
            rl = agg[sn][bi]["rougeL_f"]
            row += f" {rl:<20.3f}"
        print(row)

    # ══════════════════════════════════════════════════
    # PLOTS
    # ══════════════════════════════════════════════════
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ── Plot 1: Aggregated ROUGE-L vs budget ──
    ax = axes[0, 0]
    for sn in STRATEGIES:
        bs = [a["budget"] for a in agg[sn]]
        rl = [a["rougeL_f"] for a in agg[sn]]
        ax.plot(
            bs, rl, "-o",
            color=COLORS[sn], label=LABELS[sn],
            linewidth=2, markersize=5,
        )
    ax.set_xlabel("Budget (fraction kept)")
    ax.set_ylabel("ROUGE-L F1 (mean across prompts)")
    ax.set_title("Eviction Quality: ROUGE-L vs Budget")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    # ── Plot 2: Aggregated ROUGE-1 vs budget ──
    ax = axes[0, 1]
    for sn in STRATEGIES:
        bs = [a["budget"] for a in agg[sn]]
        r1 = [a["rouge1_f"] for a in agg[sn]]
        ax.plot(
            bs, r1, "-o",
            color=COLORS[sn], label=LABELS[sn],
            linewidth=2, markersize=5,
        )
    ax.set_xlabel("Budget (fraction kept)")
    ax.set_ylabel("ROUGE-1 F1 (mean across prompts)")
    ax.set_title("Eviction Quality: ROUGE-1 vs Budget")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    # ── Plot 3: Per-prompt ROUGE-L for sink_aware ──
    ax = axes[1, 0]
    for pi, prompt in enumerate(prompts):
        rl = [
            all_results[pi]["sink_aware"][bi]["rougeL_f"]
            for bi in range(len(budgets))
        ]
        short = prompt[:25] + "..."
        ax.plot(budgets, rl, "-o", label=short,
                markersize=4, linewidth=1.5)
    ax.set_xlabel("Budget (fraction kept)")
    ax.set_ylabel("ROUGE-L F1")
    ax.set_title("Sink-Aware: Per-Prompt Breakdown")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    # ── Plot 4: Heatmap of advantages ──
    ax = axes[1, 1]
    # Show sink_aware - uniform advantage
    advantage = np.zeros((len(STRATEGIES), len(budgets)))
    strat_names = list(STRATEGIES.keys())
    for i, sn in enumerate(strat_names):
        for j in range(len(budgets)):
            advantage[i, j] = agg[sn][j]["rougeL_f"]

    im = ax.imshow(
        advantage, cmap="RdYlGn",
        vmin=0, vmax=1, aspect="auto",
    )
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels(
        [f"{b:.0%}" for b in budgets], fontsize=7,
        rotation=45,
    )
    ax.set_yticks(range(len(strat_names)))
    ax.set_yticklabels(
        [LABELS[s] for s in strat_names], fontsize=8,
    )
    ax.set_title("ROUGE-L Heatmap (mean)")

    for i in range(len(strat_names)):
        for j in range(len(budgets)):
            val = advantage[i, j]
            c = "white" if val < 0.35 else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=7, color=c,
            )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_path = os.path.join(
        args.output_dir, "evaluation.png",
    )
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot -> {fig_path}")

    # ── Save JSON results ──
    json_path = os.path.join(
        args.output_dir, "results.json",
    )
    save_data = {
        "model": args.model,
        "video": args.video,
        "n_vis": n_vis,
        "n_sink": n_sink,
        "detect_layer": detect_layer,
        "prompts": prompts,
        "baselines": baselines,
        "budgets": budgets,
        "strategies": list(STRATEGIES.keys()),
        "aggregated": {
            sn: [
                {k: float(v) if isinstance(v, (float, np.floating)) else v
                 for k, v in a.items()}
                for a in agg[sn]
            ]
            for sn in STRATEGIES
        },
        "per_prompt": [
            {
                sn: [
                    {k: v for k, v in r.items()
                     if k != "output"}
                    for r in all_results[pi][sn]
                ]
                for sn in STRATEGIES
            }
            for pi in range(len(prompts))
        ],
    }
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"JSON -> {json_path}")

    # ── Save text summary ──
    txt_path = os.path.join(
        args.output_dir, "summary.txt",
    )
    with open(txt_path, "w") as f:
        f.write("Sink-Aware KV Cache Eviction — "
                "Systematic Evaluation\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Video: {args.video}\n")
        f.write(f"Visual: {n_vis} tokens "
                f"(sink={n_sink})\n")
        f.write(f"Prompts: {len(prompts)}\n")
        f.write(f"Budgets: {budgets}\n\n")

        f.write("AGGREGATED ROUGE-L:\n")
        f.write(f"{'Budget':<8}")
        for sn in STRATEGIES:
            f.write(f" {LABELS[sn]:<20}")
        f.write("\n" + "-" * 90 + "\n")
        for bi in range(len(budgets)):
            f.write(f"{budgets[bi]:<8.0%}")
            for sn in STRATEGIES:
                rl = agg[sn][bi]["rougeL_f"]
                f.write(f" {rl:<20.3f}")
            f.write("\n")

        f.write("\n\nPER-PROMPT BASELINES:\n")
        for pi, (prompt, bl) in enumerate(
            zip(prompts, baselines)
        ):
            f.write(f"\n  Q{pi+1}: {prompt}\n")
            f.write(f"  A: {bl[:100]}\n")

        # Find best strategy per budget
        f.write("\n\nBEST STRATEGY PER BUDGET:\n")
        for bi in range(len(budgets)):
            best_sn = max(
                STRATEGIES,
                key=lambda s: agg[s][bi]["rougeL_f"],
            )
            rl = agg[best_sn][bi]["rougeL_f"]
            f.write(
                f"  {budgets[bi]:.0%}: "
                f"{LABELS[best_sn]} "
                f"(RL={rl:.3f})\n"
            )

    print(f"Summary -> {txt_path}")

    # ── Key findings ──
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    for bi in range(len(budgets)):
        best = max(
            STRATEGIES,
            key=lambda s: agg[s][bi]["rougeL_f"],
        )
        rl = agg[best][bi]["rougeL_f"]
        sink_rl = agg["sink_aware"][bi]["rougeL_f"]
        unif_rl = agg["uniform"][bi]["rougeL_f"]
        delta = sink_rl - unif_rl
        marker = " ***" if delta > 0.03 else ""
        print(
            f"  {budgets[bi]:>4.0%}: "
            f"sink={sink_rl:.3f} "
            f"unif={unif_rl:.3f} "
            f"delta={delta:+.3f}"
            f"{marker}"
        )


if __name__ == "__main__":
    main()
