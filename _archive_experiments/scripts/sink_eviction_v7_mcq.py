"""
v7: MCQ accuracy evaluation for KV cache eviction.

Instead of ROUGE-L vs baseline (self-referential),
use ground-truth MCQ to objectively measure visual
understanding under different eviction strategies.

Key insight from v6/v6b: ROUGE-L penalizes strategies
that produce DIFFERENT but CORRECT outputs. MCQ accuracy
is a better metric.

Usage:
  uv run --with "transformers<4.52" --with torch \
    --with matplotlib \
    python sink_eviction_v7_mcq.py \
      --output-dir sink_eviction_output_v7
"""

import argparse
import json
import os
import re
from typing import List, Optional

import mlx.core as mx
import numpy as np

from sink_detect import (
    CaptureStore,
    find_image_token_range,
    patch_model,
    rmsnorm,
)


# ── Mini benchmark ───────────────────────────────────────
# Ground-truth MCQ for our test videos.
# Each entry: (video, question, choices, answer_idx)

BENCHMARK = [
    # butterfly.mp4 — pink animated butterfly with rabbit
    {
        "video": "test_videos/butterfly.mp4",
        "question": (
            "What is the main color of the butterfly's "
            "wings in this video?"
        ),
        "choices": [
            "A. Blue", "B. Pink", "C. Yellow", "D. Green",
        ],
        "answer": "B",
    },
    {
        "video": "test_videos/butterfly.mp4",
        "question": (
            "What other character appears alongside "
            "the butterfly?"
        ),
        "choices": [
            "A. A cat", "B. A dog",
            "C. A rabbit", "D. A bear",
        ],
        "answer": "C",
    },
    {
        "video": "test_videos/butterfly.mp4",
        "question": (
            "What is the dominant color of the "
            "background scenery?"
        ),
        "choices": [
            "A. Brown desert", "B. White snow",
            "C. Green vegetation", "D. Blue ocean",
        ],
        "answer": "C",
    },
    {
        "video": "test_videos/butterfly.mp4",
        "question": (
            "What style is this video rendered in?"
        ),
        "choices": [
            "A. Live-action photography",
            "B. 3D animation / cartoon",
            "C. Black and white documentary",
            "D. Watercolor painting",
        ],
        "answer": "B",
    },
    {
        "video": "test_videos/butterfly.mp4",
        "question": (
            "What is the butterfly doing in the video?"
        ),
        "choices": [
            "A. Sitting still on a flower",
            "B. Flying / fluttering",
            "C. Eating from a leaf",
            "D. Sleeping on the ground",
        ],
        "answer": "B",
    },
    # red_car.mp4 — red rectangle moving across blue bg
    {
        "video": "test_videos/red_car.mp4",
        "question": (
            "What color is the moving object in "
            "this video?"
        ),
        "choices": [
            "A. Blue", "B. Green", "C. Red", "D. White",
        ],
        "answer": "C",
    },
    {
        "video": "test_videos/red_car.mp4",
        "question": (
            "What is the background color of the video?"
        ),
        "choices": [
            "A. Red", "B. Blue", "C. Green", "D. Black",
        ],
        "answer": "B",
    },
    # blue_bounce.mp4 — blue circle bouncing
    {
        "video": "test_videos/blue_bounce.mp4",
        "question": (
            "What shape is the main object in this video?"
        ),
        "choices": [
            "A. Square", "B. Triangle",
            "C. Circle", "D. Star",
        ],
        "answer": "C",
    },
    {
        "video": "test_videos/blue_bounce.mp4",
        "question": (
            "What color is the bouncing object?"
        ),
        "choices": [
            "A. Red", "B. Blue",
            "C. Yellow", "D. Purple",
        ],
        "answer": "B",
    },
    # color_change.mp4 — background changes color
    {
        "video": "test_videos/color_change.mp4",
        "question": (
            "What happens in this video?"
        ),
        "choices": [
            "A. An object moves across the screen",
            "B. The background color changes",
            "C. Text appears on screen",
            "D. Nothing changes",
        ],
        "answer": "B",
    },
]


# ── Scoring ──────────────────────────────────────────────

def score_attn_grounding(aw, s, e, gl):
    n_vis = e - s
    scores = np.zeros(n_vis)
    n = 0
    for li in gl:
        if li >= len(aw):
            continue
        a = np.array(aw[li])
        if a.ndim == 4:
            a = a[0]
        a_mean = a.mean(axis=0)
        # All text tokens' attention to visual tokens
        seq_len = a_mean.shape[0]
        txt_mask = np.ones(seq_len, dtype=bool)
        txt_mask[s:e] = False
        txt_rows = a_mean[txt_mask]
        scores += txt_rows[:, s:e].mean(axis=0)
        n += 1
    return scores / max(n, 1)


def score_phi(hs, s, e, dims=None, layer=14):
    if dims is None:
        dims = [458, 2570]
    h = hs[min(layer, len(hs) - 1)][0]
    rms = np.abs(rmsnorm(h))
    return np.max(
        np.stack([rms[:, d] for d in dims], -1), -1,
    )[s:e]


def normalize(x):
    mn, mx_ = x.min(), x.max()
    if mx_ - mn < 1e-10:
        return np.ones_like(x) * 0.5
    return (x - mn) / (mx_ - mn)


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


def gen_evict(model, proc, ids, pv, mask,
              evict_idx, max_tok=60, **kw):
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


def extract_answer(text):
    """Extract answer letter from model output."""
    text = text.strip()
    # Try direct letter match
    m = re.search(r'\b([A-D])\b', text)
    if m:
        return m.group(1)
    # Try "Answer: X"
    m = re.search(r'[Aa]nswer[:\s]*([A-D])', text)
    if m:
        return m.group(1)
    return text[:1].upper() if text else "?"


# ── Eviction strategies ──────────────────────────────────

def make_keep_top(scores):
    """Evict lowest-scored tokens."""
    def fn(s, n, b):
        k = max(1, int(n * b))
        if k >= n:
            return None
        order = np.argsort(scores)
        return order[:n - k] + s
    return fn


def make_uniform(s_unused, n_unused=None, b_unused=None):
    pass


def strat_unif(s, n, b):
    k = max(1, int(n * b))
    return np.arange(n - k) + s if k < n else None


def strat_rand(s, n, b):
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
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument(
        "--output-dir", default="sink_eviction_output_v7",
    )
    ap.add_argument(
        "--grounding-layers", type=int, nargs="+",
        default=[18, 22, 24],
    )
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    budgets = [0.10, 0.20, 0.40, 0.60, 0.80, 1.0]
    sink_dims = [458, 2570]
    gl = args.grounding_layers

    from mlx_vlm import load
    from mlx_vlm.video_generate import process_vision_info

    print(f"Loading {args.model}...")
    model, processor = load(args.model)
    patch_model(model)

    mcfg = model.config

    def prepare(video, question, choices):
        """Build MCQ prompt."""
        choice_text = "\n".join(choices)
        prompt = (
            f"{question}\n{choice_text}\n"
            f"Answer with the letter only (A, B, C, or D)."
        )
        msgs = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video,
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
        return ids, pv, mask, extra

    # ── Per-video scoring (compute once per video) ───────
    video_scores = {}

    def get_scores(video):
        if video in video_scores:
            return video_scores[video]

        # Use a generic prompt for scoring
        prompt = "Describe what you see in this video."
        msgs = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video,
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

        s, e = find_image_token_range(ids, mcfg)
        nv = e - s

        CaptureStore.enable()
        _ = gen_evict(
            model, processor, ids, pv, mask, None,
            max_tok=20, **extra,
        )
        CaptureStore.disable()

        hs = [np.array(h) for h in CaptureStore.hidden_states]
        aw = CaptureStore.attn_weights

        sc_attn = score_attn_grounding(aw, s, e, gl)
        sc_phi = score_phi(hs, s, e, sink_dims, layer=14)
        sc_combined = normalize(sc_attn) + normalize(sc_phi)

        result = {
            "s": s, "e": e, "nv": nv,
            "attn": sc_attn,
            "phi": sc_phi,
            "combined": sc_combined,
        }
        video_scores[video] = result
        ns = int((sc_phi > 20).sum())
        print(f"  [{video}] {nv} vis tok, "
              f"{ns} sinks ({100*ns/nv:.0f}%)")
        return result

    # ── Strategies ───────────────────────────────────────
    strat_names = [
        "combined", "attn", "phi",
        "uniform", "random", "none",
    ]

    # ── Run MCQ evaluation ───────────────────────────────
    print(f"\n{'='*60}")
    print(f"MCQ EVALUATION: {len(BENCHMARK)} questions, "
          f"{len(budgets)} budgets, "
          f"{len(strat_names)} strategies")
    print(f"{'='*60}")

    # Check which videos exist
    existing = []
    for item in BENCHMARK:
        if os.path.exists(item["video"]):
            existing.append(item)
        else:
            print(f"  SKIP: {item['video']} not found")
    print(f"  {len(existing)}/{len(BENCHMARK)} "
          f"questions usable\n")

    results = {
        sn: {b: [] for b in budgets}
        for sn in strat_names
    }
    details = []

    for qi, item in enumerate(existing):
        video = item["video"]
        question = item["question"]
        choices = item["choices"]
        gt = item["answer"]

        print(f"\nQ{qi+1}: {question}")
        print(f"  GT: {gt}")

        # Get scores for this video
        vs = get_scores(video)
        s, e, nv = vs["s"], vs["e"], vs["nv"]

        # Prepare MCQ input
        ids, pv, mask, extra = prepare(
            video, question, choices,
        )

        for budget in budgets:
            for sn in strat_names:
                if sn == "none":
                    if budget < 1.0:
                        # "none" only at 100%
                        results[sn][budget].append(None)
                        continue
                    ev = None
                elif sn == "uniform":
                    ev = strat_unif(s, nv, budget)
                elif sn == "random":
                    ev = strat_rand(s, nv, budget)
                else:
                    scores = vs[sn]
                    fn = make_keep_top(scores)
                    ev = fn(s, nv, budget)

                if sn == "none" or budget == 1.0:
                    if budget == 1.0 and sn != "none":
                        # All strategies = baseline at 100%
                        results[sn][budget].append(None)
                        continue
                    # Full model, no eviction
                    out = gen_evict(
                        model, processor, ids, pv, mask,
                        None, max_tok=args.max_tokens,
                        **extra,
                    )
                else:
                    out = gen_evict(
                        model, processor, ids, pv, mask,
                        ev, max_tok=args.max_tokens,
                        **extra,
                    )

                pred = extract_answer(out)
                correct = (pred == gt)

                results[sn][budget].append(correct)
                if budget == 0.40:
                    print(f"  {sn:>10} b={budget:.0%}: "
                          f"'{out[:40]}' → {pred} "
                          f"{'✓' if correct else '✗'}")

                details.append({
                    "qi": qi,
                    "question": question,
                    "gt": gt,
                    "strategy": sn,
                    "budget": budget,
                    "pred": pred,
                    "output": out[:80],
                    "correct": correct,
                })

    # ── Aggregate accuracy ───────────────────────────────
    print(f"\n{'='*60}")
    print("ACCURACY (% correct)")
    print(f"{'='*60}")

    # Baseline accuracy (no eviction)
    bl_items = [
        d for d in details
        if d["strategy"] == "none" and d["budget"] == 1.0
    ]
    bl_acc = (
        sum(d["correct"] for d in bl_items) / len(bl_items)
        if bl_items else 0
    )
    print(f"\nBaseline (no eviction): "
          f"{bl_acc:.1%} ({sum(d['correct'] for d in bl_items)}"
          f"/{len(bl_items)})")

    print(f"\n{'Budget':<8}", end="")
    for sn in strat_names:
        if sn != "none":
            print(f"{sn:<12}", end="")
    print()
    print("-" * 68)

    acc_table = {}
    for b in budgets:
        if b == 1.0:
            continue
        print(f"{b:<8.0%}", end="")
        for sn in strat_names:
            if sn == "none":
                continue
            valid = [
                r for r in results[sn][b]
                if r is not None
            ]
            if valid:
                acc = sum(valid) / len(valid)
                acc_table.setdefault(sn, {})[b] = acc
                best = acc >= bl_acc
                print(f"{acc:<12.1%}", end="")
            else:
                print(f"{'N/A':<12}", end="")
        print()

    # ── Plot ─────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "combined": "#E91E63",
        "attn": "#2196F3",
        "phi": "#4CAF50",
        "uniform": "#FF9800",
        "random": "#9E9E9E",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_budgets = [b for b in budgets if b < 1.0]

    for sn in ["combined", "attn", "phi",
                "uniform", "random"]:
        if sn in acc_table:
            vals = [acc_table[sn].get(b, None)
                    for b in plot_budgets]
            valid_b = [b for b, v in zip(plot_budgets, vals)
                       if v is not None]
            valid_v = [v for v in vals if v is not None]
            lw = 2.5 if sn == "combined" else 1.5
            ax.plot(
                valid_b, valid_v, "-o",
                color=colors[sn], label=sn,
                linewidth=lw, markersize=6,
            )

    ax.axhline(y=bl_acc, color="black", linestyle="--",
               alpha=0.5, label=f"baseline ({bl_acc:.0%})")
    ax.set_xlabel("Budget (fraction of visual tokens kept)")
    ax.set_ylabel("MCQ Accuracy")
    ax.set_title(
        "v7: KV Cache Eviction — MCQ Accuracy\n"
        f"{len(existing)} questions, ground-truth evaluation"
    )
    ax.legend()
    ax.set_xlim(0, 0.85)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fp = os.path.join(args.output_dir, "v7_accuracy.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot -> {fp}")

    # Save JSON
    jp = os.path.join(args.output_dir, "results.json")
    with open(jp, "w") as f:
        json.dump({
            "baseline_acc": bl_acc,
            "n_questions": len(existing),
            "budgets": budgets,
            "accuracy": {
                sn: {
                    str(b): acc_table.get(sn, {}).get(b)
                    for b in plot_budgets
                }
                for sn in strat_names if sn != "none"
            },
            "details": details,
        }, f, indent=2)
    print(f"JSON -> {jp}")


if __name__ == "__main__":
    main()
