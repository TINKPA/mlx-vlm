"""
v8: Multi-turn QA after eviction.

v7 showed prefill→evict→decode preserves factual accuracy
because visual info is already encoded in text hidden states.

This experiment tests a HARDER scenario: prefill → evict →
answer Q1 → answer Q2 → ... → answer Qn, where each question
is asked in a NEW forward pass that must attend to the
(now-evicted) visual KV cache.

This simulates streaming VLM: the visual KV cache is shared
across multiple queries, and eviction degrades the cache.

Usage:
  uv run --with "transformers<4.52" --with torch \
    --with matplotlib \
    python sink_eviction_v8_multiturn.py \
      --output-dir sink_eviction_output_v8
"""

import argparse
import json
import os
import re

import mlx.core as mx
import numpy as np

from sink_detect import (
    CaptureStore,
    find_image_token_range,
    patch_model,
    rmsnorm,
)


# ── MCQ Benchmark ────────────────────────────────────────

BENCHMARK = {
    "test_videos/butterfly.mp4": [
        {
            "question": (
                "What is the main color of the butterfly's "
                "wings?"
            ),
            "choices": [
                "A. Blue", "B. Pink",
                "C. Yellow", "D. Green",
            ],
            "answer": "B",
            "difficulty": "easy",
        },
        {
            "question": (
                "What other character appears alongside "
                "the butterfly?"
            ),
            "choices": [
                "A. A cat", "B. A dog",
                "C. A rabbit", "D. A bear",
            ],
            "answer": "C",
            "difficulty": "easy",
        },
        {
            "question": (
                "What is the dominant color of the "
                "background?"
            ),
            "choices": [
                "A. Brown", "B. White",
                "C. Green", "D. Blue ocean",
            ],
            "answer": "C",
            "difficulty": "easy",
        },
        {
            "question": (
                "What style is this video?"
            ),
            "choices": [
                "A. Live-action",
                "B. 3D animation / cartoon",
                "C. Black and white",
                "D. Watercolor painting",
            ],
            "answer": "B",
            "difficulty": "medium",
        },
        {
            "question": (
                "What is the butterfly doing?"
            ),
            "choices": [
                "A. Sitting still",
                "B. Flying / fluttering",
                "C. Eating", "D. Sleeping",
            ],
            "answer": "B",
            "difficulty": "easy",
        },
        {
            "question": (
                "What kind of day is shown in the video?"
            ),
            "choices": [
                "A. Rainy day",
                "B. Nighttime",
                "C. Bright sunny day",
                "D. Foggy morning",
            ],
            "answer": "C",
            "difficulty": "medium",
        },
        {
            "question": (
                "Are there flowers visible in the scene?"
            ),
            "choices": [
                "A. Yes, many flowers",
                "B. No flowers at all",
                "C. Only one flower",
                "D. Cannot determine",
            ],
            "answer": "A",
            "difficulty": "medium",
        },
    ],
}


# ── Scoring ──────────────────────────────────────────────

def score_attn(aw, s, e, gl):
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
        seq_len = a_mean.shape[0]
        txt_mask = np.ones(seq_len, dtype=bool)
        txt_mask[s:e] = False
        scores += a_mean[txt_mask][:, s:e].mean(axis=0)
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


# ── KV cache operations ──────────────────────────────────

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


def prefill_and_evict(model, proc, ids, pv, mask,
                      evict_idx, **kw):
    """Prefill, optionally evict, return shared cache."""
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

    if evict_idx is not None and len(evict_idx) > 0:
        sl = cache[0].offset
        es = set(evict_idx.tolist())
        keep = np.array(
            [i for i in range(sl) if i not in es],
            dtype=np.int32,
        )
        evict_kv(cache, keep)

    return cache


def answer_from_cache(model, proc, cache, question_text,
                      max_tok=30):
    """
    Encode a follow-up question and decode answer,
    reusing the existing KV cache.

    This simulates multi-turn: the visual KV is shared.
    """
    # Tokenize the question
    tokenizer = getattr(proc, 'tokenizer', proc)
    q_ids = tokenizer.encode(
        "\n" + question_text + "\nAnswer with the letter "
        "only (A, B, C, or D).",
        add_special_tokens=False,
    )
    q_ids_mx = mx.array([q_ids])

    # Forward through LM with shared cache
    out = model.language_model(q_ids_mx, cache=cache)
    mx.eval([c.state for c in cache])
    logits = out.logits[:, -1, :]

    # Greedy decode
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
    text = text.strip()
    m = re.search(r'\b([A-D])\b', text)
    if m:
        return m.group(1)
    m = re.search(r'[Aa]nswer[:\s]*([A-D])', text)
    if m:
        return m.group(1)
    return text[:1].upper() if text else "?"


# ── Eviction strategies ──────────────────────────────────

def make_evict_keep_top(scores, s, n, b):
    k = max(1, int(n * b))
    if k >= n:
        return None
    order = np.argsort(scores)
    return order[:n - k] + s


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
    ap.add_argument("--max-tokens", type=int, default=30)
    ap.add_argument(
        "--output-dir", default="sink_eviction_output_v8",
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

    strat_names = [
        "combined", "phi", "attn",
        "uniform", "random",
    ]

    for video, questions in BENCHMARK.items():
        if not os.path.exists(video):
            print(f"SKIP: {video}")
            continue

        print(f"\n{'#'*60}")
        print(f"VIDEO: {video} ({len(questions)} questions)")
        print(f"{'#'*60}")

        # ── Prepare video input ──────────────────────────
        prompt_base = "Watch this video carefully."
        msgs = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video,
                    "max_pixels": 224 * 224,
                    "fps": 1.0,
                },
                {"type": "text", "text": prompt_base},
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
        print(f"Visual tokens: {nv}")

        # ── Compute scores ───────────────────────────────
        CaptureStore.enable()
        cache_tmp = prefill_and_evict(
            model, processor, ids, pv, mask,
            None, **extra,
        )
        CaptureStore.disable()

        hs = [np.array(h) for h in CaptureStore.hidden_states]
        aw = CaptureStore.attn_weights

        sc_attn = score_attn(aw, s, e, gl)
        sc_phi = score_phi(hs, s, e, sink_dims)
        sc_combined = normalize(sc_attn) + normalize(sc_phi)

        ns = int((sc_phi > 20).sum())
        print(f"Sinks: {ns}/{nv} ({100*ns/nv:.0f}%)")

        # ── Multi-turn evaluation ────────────────────────
        all_results = []

        for budget in budgets:
            print(f"\n--- Budget: {budget:.0%} ---")

            for sn in strat_names:
                # Compute eviction indices
                if sn == "combined":
                    ev = make_evict_keep_top(
                        sc_combined, s, nv, budget,
                    )
                elif sn == "phi":
                    ev = make_evict_keep_top(
                        sc_phi, s, nv, budget,
                    )
                elif sn == "attn":
                    ev = make_evict_keep_top(
                        sc_attn, s, nv, budget,
                    )
                elif sn == "uniform":
                    ev = strat_unif(s, nv, budget)
                elif sn == "random":
                    ev = strat_rand(s, nv, budget)

                # Prefill + evict once
                cache = prefill_and_evict(
                    model, processor, ids, pv, mask,
                    ev, **extra,
                )

                # Ask all questions sequentially on shared cache
                correct = 0
                for qi, q in enumerate(questions):
                    q_text = (
                        q["question"] + "\n"
                        + "\n".join(q["choices"])
                    )
                    out = answer_from_cache(
                        model, processor, cache,
                        q_text, max_tok=args.max_tokens,
                    )
                    pred = extract_answer(out)
                    is_correct = (pred == q["answer"])
                    correct += is_correct

                    all_results.append({
                        "video": video,
                        "budget": budget,
                        "strategy": sn,
                        "qi": qi,
                        "question": q["question"],
                        "gt": q["answer"],
                        "pred": pred,
                        "output": out[:50],
                        "correct": is_correct,
                        "turn": qi + 1,
                    })

                acc = correct / len(questions)
                tag = "★" if acc > 0.7 else " "
                print(f"  {sn:<12} {acc:.0%} "
                      f"({correct}/{len(questions)}) {tag}")

    # ── Aggregate ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")

    # Per strategy x budget
    acc_table = {}
    for sn in strat_names:
        acc_table[sn] = {}
        for b in budgets:
            items = [
                r for r in all_results
                if r["strategy"] == sn
                and r["budget"] == b
            ]
            if items:
                acc_table[sn][b] = (
                    sum(r["correct"] for r in items)
                    / len(items)
                )

    print(f"\n{'Budget':<8}", end="")
    for sn in strat_names:
        print(f"{sn:<12}", end="")
    print()
    print("-" * 68)
    for b in budgets:
        print(f"{b:<8.0%}", end="")
        for sn in strat_names:
            v = acc_table.get(sn, {}).get(b)
            if v is not None:
                print(f"{v:<12.0%}", end="")
            else:
                print(f"{'N/A':<12}", end="")
        print()

    # Per turn (does accuracy degrade with more turns?)
    print(f"\nAccuracy by turn (budget=40%):")
    for sn in strat_names:
        items_40 = [
            r for r in all_results
            if r["strategy"] == sn
            and abs(r["budget"] - 0.40) < 0.01
        ]
        turns = {}
        for r in items_40:
            t = r["turn"]
            turns.setdefault(t, []).append(r["correct"])
        if turns:
            print(f"  {sn:<12}", end="")
            for t in sorted(turns.keys()):
                acc = sum(turns[t]) / len(turns[t])
                print(f"T{t}:{acc:.0%} ", end="")
            print()

    # ── Plot ─────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "combined": "#E91E63", "attn": "#2196F3",
        "phi": "#4CAF50", "uniform": "#FF9800",
        "random": "#9E9E9E",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for sn in strat_names:
        if sn in acc_table:
            bs = sorted(acc_table[sn].keys())
            vals = [acc_table[sn][b] for b in bs]
            lw = 2.5 if sn == "combined" else 1.5
            ax.plot(
                bs, vals, "-o", color=colors[sn],
                label=sn, linewidth=lw, markersize=6,
            )
    ax.set_xlabel("Budget (fraction of visual tokens kept)")
    ax.set_ylabel("MCQ Accuracy")
    ax.set_title(
        "v8: Multi-Turn QA After KV Cache Eviction\n"
        "Prefill once → Evict → Answer multiple questions"
    )
    ax.legend()
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fp = os.path.join(args.output_dir, "v8_multiturn.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot -> {fp}")

    # Save JSON
    jp = os.path.join(args.output_dir, "results.json")
    with open(jp, "w") as f:
        json.dump({
            "budgets": budgets,
            "strategies": strat_names,
            "accuracy": {
                sn: {
                    str(b): acc_table.get(sn, {}).get(b)
                    for b in budgets
                }
                for sn in strat_names
            },
            "details": all_results,
        }, f, indent=2)
    print(f"JSON -> {jp}")


if __name__ == "__main__":
    main()
