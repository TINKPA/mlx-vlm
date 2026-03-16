"""
v6b: Combined scoring for KV cache eviction.

Key insight from v6: attn and φ correlate at only 0.195,
meaning they capture different aspects of token importance:
  - φ: structural anchors (position-dependent sinks)
  - attn: evidence-relevant tokens (content-dependent)

This version tries:
  1. combined = normalize(attn) + normalize(φ)
  2. attn with VEA-style denoising (neighborhood filter)
  3. All-text-token attention (not just last token)

Also uses more prompts for less noisy results.

Usage:
  uv run --with "transformers<4.52" --with torch \
    --with matplotlib --with rouge-score \
    python sink_eviction_v6b_combined.py \
      --video test_videos/butterfly.mp4 \
      --output-dir sink_eviction_output_v6b
"""

import argparse
import json
import os
from typing import List

import mlx.core as mx
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


def normalize(x):
    """Min-max normalize to [0, 1]."""
    mn, mx_ = x.min(), x.max()
    if mx_ - mn < 1e-10:
        return np.ones_like(x) * 0.5
    return (x - mn) / (mx_ - mn)


# ── Scoring functions ────────────────────────────────────

def score_attn_last_tok(aw_list, s, e, gl):
    """Attention from LAST token to visual tokens."""
    n_vis = e - s
    scores = np.zeros(n_vis)
    n = 0
    for li in gl:
        if li >= len(aw_list):
            continue
        aw = np.array(aw_list[li])
        if aw.ndim == 4:
            aw = aw[0]
        aw_mean = aw.mean(axis=0)
        scores += aw_mean[-1, s:e]
        n += 1
    return scores / max(n, 1)


def score_attn_all_text(aw_list, s, e, gl):
    """Attention from ALL text tokens to visual tokens.
    Averages over text query positions and grounding layers.
    """
    n_vis = e - s
    scores = np.zeros(n_vis)
    n = 0
    for li in gl:
        if li >= len(aw_list):
            continue
        aw = np.array(aw_list[li])
        if aw.ndim == 4:
            aw = aw[0]
        aw_mean = aw.mean(axis=0)  # [Q, K]
        seq_len = aw_mean.shape[0]
        # Text tokens = all non-visual
        txt_mask = np.ones(seq_len, dtype=bool)
        txt_mask[s:e] = False
        txt_rows = aw_mean[txt_mask]  # [n_txt, K]
        scores += txt_rows[:, s:e].mean(axis=0)  # [n_vis]
        n += 1
    return scores / max(n, 1)


def score_attn_denoised(aw_list, s, e, gl, grid_hw=None,
                         lam=10.0):
    """
    Attention with VEA-style neighborhood denoising.
    If a token's score > λ * max(neighbors), replace with
    neighbor average. Requires grid_hw for 2D layout.
    """
    raw = score_attn_all_text(aw_list, s, e, gl)
    if grid_hw is None:
        return raw  # can't denoise without spatial layout

    H, W = grid_hw
    n_vis = e - s
    # May have multiple frames
    n_per_frame = H * W
    n_frames = n_vis // n_per_frame
    if n_frames == 0:
        return raw

    denoised = raw.copy()
    for fi in range(n_frames):
        start = fi * n_per_frame
        end = start + n_per_frame
        if end > n_vis:
            break
        grid = raw[start:end].reshape(H, W)
        out = grid.copy()

        for i in range(H):
            for j in range(W):
                # 3x3 neighborhood
                i_lo = max(0, i - 1)
                i_hi = min(H, i + 2)
                j_lo = max(0, j - 1)
                j_hi = min(W, j + 2)
                nb = grid[i_lo:i_hi, j_lo:j_hi]
                # Exclude self
                nb_flat = []
                for di in range(i_lo, i_hi):
                    for dj in range(j_lo, j_hi):
                        if di != i or dj != j:
                            nb_flat.append(grid[di, dj])
                if not nb_flat:
                    continue
                nb_arr = np.array(nb_flat)
                nb_max = nb_arr.max()
                nb_avg = nb_arr.mean()
                # If score > λ * max(neighbors), it's a spike
                if grid[i, j] > lam * nb_max and nb_max > 0:
                    out[i, j] = nb_avg

        denoised[start:end] = out.flatten()
    return denoised


def score_phi(hs, s, e, dims=None, layer=14):
    if dims is None:
        dims = [458, 2570]
    h = hs[min(layer, len(hs) - 1)][0]
    rms = np.abs(rmsnorm(h))
    phi = np.max(
        np.stack([rms[:, d] for d in dims], -1), -1,
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


def gen_evict(model, proc, text, ids, pv, mask,
              evict_idx, max_tok=80, **kw):
    from mlx_vlm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model.language_model)
    eo = model.get_input_embeddings(ids, pv, mask=mask, **kw)
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

def make_evict_fn(ascending=True):
    """Generic: keep top-k by score. ascending=True means
    higher score = more important = keep."""
    def fn(scores, s, n, b):
        k = max(1, int(n * b))
        if k >= n:
            return None
        if ascending:
            order = np.argsort(scores)  # low first
            return order[:n - k] + s    # evict lowest
        else:
            order = np.argsort(scores)[::-1]  # high first
            return order[k:] + s              # evict past k
    return fn


def strat_unif(scores, s, n, b):
    k = max(1, int(n * b))
    return np.arange(n - k) + s if k < n else None


def strat_rand(scores, s, n, b):
    k = max(1, int(n * b))
    if k >= n:
        return None
    return np.random.RandomState(42).choice(
        n, n - k, replace=False,
    ) + s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-VL-7B-Instruct-8bit",
    )
    ap.add_argument("--video", required=True)
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument(
        "--output-dir", default="sink_eviction_output_v6b",
    )
    ap.add_argument(
        "--grounding-layers", type=int, nargs="+",
        default=[18, 22, 24],
    )
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    budgets = [0.08, 0.20, 0.40, 0.60, 0.80, 1.0]
    sink_dims = [458, 2570]
    gl = args.grounding_layers

    prompts = [
        "Describe the butterfly in this video.",
        "What color are the butterfly's wings?",
        "What is the background scenery?",
        "What is happening in this video?",
        "Describe the motion of the butterfly.",
        "How many butterflies are in the video?",
        "What type of environment is shown?",
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
        grid_hw = None
        for k in ["video_grid_thw", "image_grid_thw"]:
            if inputs.get(k) is not None:
                extra[k] = mx.array(inputs[k])
                thw = np.array(inputs[k][0])
                sms = model.config.vision_config\
                    .spatial_merge_size
                grid_hw = (
                    int(thw[1]) // sms,
                    int(thw[2]) // sms,
                )
        return text, ids, pv, mask, extra, grid_hw

    # ── Capture ──────────────────────────────────────────
    print("\nCapturing attention + hidden states...")
    t0, ids0, pv0, m0, e0, grid_hw = prepare(prompts[0])
    mcfg = model.config
    s, e = find_image_token_range(ids0, mcfg)
    nv = e - s
    print(f"Visual tokens: {nv}, grid: {grid_hw}")

    CaptureStore.enable()
    _ = gen_evict(
        model, processor, t0,
        ids0, pv0, m0, None,
        max_tok=args.max_tokens, **e0,
    )
    CaptureStore.disable()

    hs = [np.array(h) for h in CaptureStore.hidden_states]
    aw = CaptureStore.attn_weights

    # ── Compute all scoring variants ─────────────────────
    print("\nComputing scores...")

    sc_attn_last = score_attn_last_tok(aw, s, e, gl)
    sc_attn_all = score_attn_all_text(aw, s, e, gl)
    sc_attn_dn = score_attn_denoised(
        aw, s, e, gl, grid_hw, lam=10.0,
    )
    sc_phi = score_phi(hs, s, e, sink_dims, layer=14)

    # Combined: normalize + add
    sc_combined = normalize(sc_attn_all) + normalize(sc_phi)
    sc_combined_dn = (
        normalize(sc_attn_dn) + normalize(sc_phi)
    )

    # Print stats & correlations
    print(f"\nScore correlations:")
    labels = [
        "attn_last", "attn_all", "attn_dn",
        "phi", "combined", "combined_dn",
    ]
    all_sc = [
        sc_attn_last, sc_attn_all, sc_attn_dn,
        sc_phi, sc_combined, sc_combined_dn,
    ]
    for i, (l1, s1) in enumerate(zip(labels, all_sc)):
        for j, (l2, s2) in enumerate(zip(labels, all_sc)):
            if j > i:
                c = np.corrcoef(s1, s2)[0, 1]
                if abs(c) > 0.3:
                    print(f"  {l1} vs {l2}: {c:.3f}")

    ns_phi = int((sc_phi > 20).sum())
    print(f"\nφ sinks: {ns_phi}/{nv} ({100*ns_phi/nv:.0f}%)")

    # ── Strategies ───────────────────────────────────────
    keep_high = make_evict_fn(ascending=True)

    strats = {
        "combined": (sc_combined, keep_high),
        "combined_dn": (sc_combined_dn, keep_high),
        "attn_all": (sc_attn_all, keep_high),
        "attn_denoised": (sc_attn_dn, keep_high),
        "phi_based": (sc_phi, keep_high),
        "uniform": (sc_phi, strat_unif),
        "random": (sc_phi, strat_rand),
    }

    colors = {
        "combined": "#E91E63",
        "combined_dn": "#9C27B0",
        "attn_all": "#2196F3",
        "attn_denoised": "#00BCD4",
        "phi_based": "#4CAF50",
        "uniform": "#FF9800",
        "random": "#9E9E9E",
    }

    # ── Run experiments ──────────────────────────────────
    rl_data = {
        sn: {b: [] for b in budgets}
        for sn in strats
    }

    for pi, prompt in enumerate(prompts):
        tp, idsp, pvp, mp, ep, _ = prepare(prompt)
        bl = gen_evict(
            model, processor, tp,
            idsp, pvp, mp, None,
            max_tok=args.max_tokens, **ep,
        )
        print(f"\nP{pi+1}: {bl[:50]}")

        for sn, (scores, sfn) in strats.items():
            for budget in budgets:
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
                rl_data[sn][budget].append(
                    compute_rl(bl, out)
                )

    # ── Results ──────────────────────────────────────────
    agg = {
        sn: {b: np.mean(rl_data[sn][b]) for b in budgets}
        for sn in strats
    }

    print("\n" + "=" * 80)
    print("RESULTS: ROUGE-L (avg over prompts)")
    print("=" * 80)

    print(f"\n{'Budget':<8}", end="")
    for sn in strats:
        print(f"{sn:<14}", end="")
    print()
    print("-" * 106)
    for b in budgets:
        print(f"{b:<8.0%}", end="")
        for sn in strats:
            print(f"{agg[sn][b]:<14.3f}", end="")
        print()

    # Delta vs uniform
    print(f"\n{'Budget':<8}", end="")
    for sn in strats:
        if sn != "uniform":
            print(f"{sn:<14}", end="")
    print()
    print("-" * 92)
    for b in budgets:
        print(f"{b:<8.0%}", end="")
        for sn in strats:
            if sn != "uniform":
                d = agg[sn][b] - agg["uniform"][b]
                m = "*" if d > 0.03 else " "
                print(f"{d:>+6.3f}{m}       ", end="")
        print()

    # ── Plot ─────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for sn in strats:
        vals = [agg[sn][b] for b in budgets]
        ls = "-" if sn in (
            "combined", "combined_dn", "uniform",
        ) else "--"
        lw = 2.5 if sn in (
            "combined", "combined_dn",
        ) else 1.5
        ax.plot(
            budgets, vals, f"{ls}o",
            color=colors[sn], label=sn,
            linewidth=lw, markersize=4,
        )
    ax.set_xlabel("Budget (fraction kept)")
    ax.set_ylabel("ROUGE-L F1")
    ax.set_title(
        f"v6b: Combined Scoring for KV Cache Eviction\n"
        f"{nv} visual tokens, {len(prompts)} prompts, "
        f"grounding layers {gl}"
    )
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fp = os.path.join(args.output_dir, "v6b_comparison.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot -> {fp}")

    # Save JSON
    jp = os.path.join(args.output_dir, "results.json")
    with open(jp, "w") as f:
        json.dump({
            "n_vis": nv,
            "grounding_layers": gl,
            "n_prompts": len(prompts),
            "budgets": budgets,
            "agg": {
                sn: {
                    str(b): float(v)
                    for b, v in agg[sn].items()
                }
                for sn in strats
            },
        }, f, indent=2)
    print(f"JSON -> {jp}")


if __name__ == "__main__":
    main()
