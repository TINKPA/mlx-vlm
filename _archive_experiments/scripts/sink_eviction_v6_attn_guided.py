"""
v6: Attention-guided KV cache eviction.

Inspired by "Seeing but Not Believing" (ICLR 2026):
  - Use attention from visual grounding layers {18, 22, 24}
    to score visual token importance
  - Keep tokens with highest attention scores, evict rest
  - Compare with φ-based, uniform, and random eviction

Key insight: grounding-layer attention (AUROC 89% on Qwen2.5-VL)
is a better importance signal than φ (DimProspector), which
over-detects sinks (80-92%) and doesn't transfer across
resolutions.

Usage:
  uv run --with "transformers<4.52" --with torch \
    --with matplotlib --with rouge-score \
    python sink_eviction_v6_attn_guided.py \
      --video test_videos/butterfly.mp4 \
      --output-dir sink_eviction_output_v6
"""

import argparse
import json
import os
from typing import List, Optional

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


# ── Attention-based importance scoring ───────────────────

def score_attn_grounding(
    attn_weights: List[np.ndarray],
    s: int, e: int,
    grounding_layers: List[int] = None,
) -> np.ndarray:
    """
    Score visual tokens by attention they receive from the
    LAST token (first generated token position) across
    visual grounding layers.

    Returns importance scores for visual tokens [s:e].
    Higher = more important = keep.
    """
    if grounding_layers is None:
        # Qwen2.5-VL-7B grounding layers from VEA paper
        grounding_layers = [18, 22, 24]

    n_vis = e - s
    scores = np.zeros(n_vis)
    n_valid = 0

    for li in grounding_layers:
        if li >= len(attn_weights):
            continue
        # attn: [1, n_heads, seq_len, seq_len]
        aw = np.array(attn_weights[li])
        if aw.ndim == 4:
            aw = aw[0]  # [n_heads, Q, K]
        # Average across heads
        aw_mean = aw.mean(axis=0)  # [Q, K]
        # Last token's attention to visual tokens
        last_tok_attn = aw_mean[-1, s:e]  # [n_vis]
        scores += last_tok_attn
        n_valid += 1

    if n_valid > 0:
        scores /= n_valid

    return scores


def score_phi(
    hidden_states: List[np.ndarray],
    s: int, e: int,
    sink_dims: List[int] = None,
    layer: int = 14,
    tau: float = 20.0,
) -> np.ndarray:
    """
    Score visual tokens by φ (DimProspector).
    Higher φ = more "sinky" = structural anchor.
    """
    if sink_dims is None:
        sink_dims = [458, 2570]
    h = hidden_states[min(layer, len(hidden_states) - 1)][0]
    rms = np.abs(rmsnorm(h))
    phi = np.max(
        np.stack([rms[:, d] for d in sink_dims], -1), -1,
    )
    return phi[s:e]


# ── KV cache eviction ────────────────────────────────────

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


# ── Eviction strategies ──────────────────────────────────

def strat_attn(scores, s, n, b):
    """Keep tokens with highest attention scores."""
    k = max(1, int(n * b))
    if k >= n:
        return None
    # Argsort ascending, evict the lowest-scored
    order = np.argsort(scores)  # ascending
    evict_local = order[:n - k]  # lowest scores
    return evict_local + s


def strat_phi(scores, s, n, b):
    """Keep tokens with highest φ (sink = anchor)."""
    k = max(1, int(n * b))
    if k >= n:
        return None
    order = np.argsort(scores)[::-1]  # descending
    evict_local = order[k:]  # lowest φ
    return evict_local + s


def strat_unif(scores, s, n, b):
    """Evict first (1-b)*n tokens (oldest)."""
    k = max(1, int(n * b))
    if k >= n:
        return None
    return np.arange(n - k) + s


def strat_rand(scores, s, n, b):
    """Random eviction."""
    k = max(1, int(n * b))
    if k >= n:
        return None
    return np.random.RandomState(42).choice(
        n, n - k, replace=False,
    ) + s


STRATS = {
    "attn_guided": strat_attn,
    "phi_based": strat_phi,
    "uniform": strat_unif,
    "random": strat_rand,
}
COLORS = {
    "attn_guided": "#2196F3",
    "phi_based": "#4CAF50",
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
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument(
        "--output-dir",
        default="sink_eviction_output_v6",
    )
    parser.add_argument(
        "--grounding-layers", type=int, nargs="+",
        default=[18, 22, 24],
        help="Grounding layers from VEA paper profiling",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    budgets = [0.08, 0.20, 0.40, 0.60, 0.80, 1.0]
    sink_dims = [458, 2570]

    prompts = [
        "Describe the butterfly in this video.",
        "What color are the butterfly's wings?",
        "What is the background scenery?",
        "What is happening in this video?",
        "Describe the motion of the butterfly.",
    ]

    from mlx_vlm import load
    from mlx_vlm.video_generate import process_vision_info

    print(f"Loading {args.model}...")
    model, processor = load(args.model)
    patch_model(model)

    def prepare(prompt):
        msgs = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": args.video,
                    "max_pixels": 224 * 224,
                    "fps": 1.0,
                },
                {"type": "text", "text": prompt},
            ],
        }]
        text = processor.apply_chat_template(
            msgs, tokenize=False,
            add_generation_prompt=True,
        )
        img_in, vid_in, _ = process_vision_info(msgs, True)
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

    # ── Step 1: Capture attention from grounding layers ──
    print("\n" + "=" * 60)
    print("STEP 1: Capture attention + hidden states")
    print("=" * 60)

    t0, ids0, pv0, m0, e0 = prepare(prompts[0])
    mcfg = model.config
    s, e = find_image_token_range(ids0, mcfg)
    nv = e - s
    print(f"Visual tokens: {nv} [{s}:{e}]")
    print(f"Grounding layers: {args.grounding_layers}")

    CaptureStore.enable()
    _ = gen_evict(
        model, processor, t0,
        ids0, pv0, m0, None,
        max_tok=args.max_tokens, **e0,
    )
    CaptureStore.disable()

    hs = [np.array(h) for h in CaptureStore.hidden_states]
    aw = CaptureStore.attn_weights  # keep as mx for now

    print(f"Captured {len(hs)} hidden states, "
          f"{len(aw)} attention weights")

    # ── Step 2: Compute importance scores ────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Compute importance scores")
    print("=" * 60)

    attn_scores = score_attn_grounding(
        aw, s, e, args.grounding_layers,
    )
    phi_scores = score_phi(hs, s, e, sink_dims, layer=14)

    # Stats
    print(f"\nAttention scores (grounding layers):")
    print(f"  mean={attn_scores.mean():.6f}, "
          f"std={attn_scores.std():.6f}")
    print(f"  min={attn_scores.min():.6f}, "
          f"max={attn_scores.max():.6f}")

    print(f"\nφ scores (DimProspector):")
    print(f"  mean={phi_scores.mean():.2f}, "
          f"std={phi_scores.std():.2f}")
    ns_phi = int((phi_scores > 20).sum())
    print(f"  sinks (φ>20): {ns_phi}/{nv} "
          f"({100*ns_phi/nv:.0f}%)")

    # Correlation between the two scoring methods
    corr = np.corrcoef(attn_scores, phi_scores)[0, 1]
    print(f"\nCorrelation(attn, φ): {corr:.3f}")

    # ── Step 3: Run eviction experiments ─────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Eviction experiments")
    print("=" * 60)

    # Score lookup for strategies
    score_map = {
        "attn_guided": attn_scores,
        "phi_based": phi_scores,
        "uniform": phi_scores,  # unused by uniform
        "random": phi_scores,   # unused by random
    }

    rl_data = {
        sn: {b: [] for b in budgets}
        for sn in STRATS
    }

    for pi, prompt in enumerate(prompts):
        tp, idsp, pvp, mp, ep = prepare(prompt)
        bl = gen_evict(
            model, processor, tp,
            idsp, pvp, mp, None,
            max_tok=args.max_tokens, **ep,
        )
        print(f"\nP{pi+1}: {prompt}")
        print(f"  BL: {bl[:60]}")

        for sn, sfn in STRATS.items():
            for budget in budgets:
                scores = score_map[sn]
                ev = sfn(scores, s, nv, budget)
                if ev is None:
                    out = bl
                else:
                    out = gen_evict(
                        model, processor, tp,
                        idsp, pvp, mp, ev,
                        max_tok=args.max_tokens,
                        **ep,
                    )
                rl = compute_rl(bl, out)
                rl_data[sn][budget].append(rl)

    # ── Step 4: Results ──────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS: ROUGE-L (avg over prompts)")
    print("=" * 60)

    agg = {}
    for sn in STRATS:
        agg[sn] = {
            b: np.mean(rl_data[sn][b])
            for b in budgets
        }

    print(f"\n{'Budget':<8} ", end="")
    for sn in STRATS:
        print(f"{sn:<14} ", end="")
    print()
    print("-" * 70)
    for b in budgets:
        print(f"{b:<8.0%} ", end="")
        for sn in STRATS:
            print(f"{agg[sn][b]:<14.3f} ", end="")
        print()

    # Delta table
    print(f"\n{'Budget':<8} attn-unif   phi-unif    "
          f"attn-phi")
    print("-" * 50)
    for b in budgets:
        au = agg["attn_guided"][b] - agg["uniform"][b]
        pu = agg["phi_based"][b] - agg["uniform"][b]
        ap = agg["attn_guided"][b] - agg["phi_based"][b]
        print(f"{b:<8.0%} {au:>+8.3f}    "
              f"{pu:>+8.3f}    {ap:>+8.3f}")

    # ── Step 5: Plot ─────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: ROUGE-L curves
    for sn in STRATS:
        vals = [agg[sn][b] for b in budgets]
        ax1.plot(
            budgets, vals, "-o",
            color=COLORS[sn], label=sn,
            linewidth=2, markersize=5,
        )
    ax1.set_xlabel("Budget (fraction kept)")
    ax1.set_ylabel("ROUGE-L F1")
    ax1.set_title(
        f"Eviction Strategy Comparison\n"
        f"({nv} visual tokens, "
        f"grounding layers {args.grounding_layers})"
    )
    ax1.legend()
    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3)

    # Plot 2: Score distributions
    ax2.hist(
        attn_scores, bins=50, alpha=0.6,
        color="#2196F3", label="attn scores",
        density=True,
    )
    ax2_twin = ax2.twinx()
    ax2_twin.hist(
        phi_scores, bins=50, alpha=0.6,
        color="#4CAF50", label="φ scores",
        density=True,
    )
    ax2.set_xlabel("Score")
    ax2.set_ylabel("Density (attn)", color="#2196F3")
    ax2_twin.set_ylabel("Density (φ)", color="#4CAF50")
    ax2.set_title(
        f"Score Distributions\n"
        f"corr(attn, φ) = {corr:.3f}"
    )
    ax2.legend(loc="upper left")
    ax2_twin.legend(loc="upper right")

    plt.tight_layout()
    fp = os.path.join(args.output_dir, "v6_comparison.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot -> {fp}")

    # Delta bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(budgets))
    w = 0.25
    for i, (sn, color) in enumerate([
        ("attn_guided", "#2196F3"),
        ("phi_based", "#4CAF50"),
        ("random", "#9E9E9E"),
    ]):
        deltas = [
            agg[sn][b] - agg["uniform"][b]
            for b in budgets
        ]
        ax.bar(
            x + (i - 1) * w, deltas, w,
            label=f"{sn} - uniform",
            color=color, alpha=0.8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:.0%}" for b in budgets])
    ax.set_xlabel("Budget")
    ax.set_ylabel("ROUGE-L Delta vs Uniform")
    ax.set_title(
        "Attention-Guided vs φ-Based vs Random\n"
        "(relative to uniform baseline)"
    )
    ax.axhline(y=0, color="black", lw=0.8)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fp2 = os.path.join(args.output_dir, "v6_delta.png")
    fig.savefig(fp2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Delta plot -> {fp2}")

    # Save JSON
    jp = os.path.join(args.output_dir, "results.json")
    with open(jp, "w") as f:
        json.dump({
            "n_vis": nv,
            "grounding_layers": args.grounding_layers,
            "sink_dims": sink_dims,
            "n_sink_phi": ns_phi,
            "corr_attn_phi": float(corr),
            "budgets": budgets,
            "agg": {
                sn: {
                    str(b): float(v)
                    for b, v in agg[sn].items()
                }
                for sn in STRATS
            },
            "attn_scores_stats": {
                "mean": float(attn_scores.mean()),
                "std": float(attn_scores.std()),
                "min": float(attn_scores.min()),
                "max": float(attn_scores.max()),
            },
            "phi_scores_stats": {
                "mean": float(phi_scores.mean()),
                "std": float(phi_scores.std()),
                "n_sink": ns_phi,
            },
        }, f, indent=2)
    print(f"JSON -> {jp}")


if __name__ == "__main__":
    main()
