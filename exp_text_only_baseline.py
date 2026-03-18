"""
Text-only baseline: answer MVBench MCQ without any video input.

If text-only accuracy ≈ full-model accuracy, then the model
is not using visual information on this benchmark.

Usage:
  uv run --with "transformers<4.52" --with torch \
    python exp_text_only_baseline.py \
      --data-dir /Volumes/RAID0/datasets/MVBench \
      --max-questions 260 \
      --output-dir exp_text_only_output
"""

import argparse
import json
import os
import time

import mlx.core as mx
import numpy as np

from sink_eviction_v10_mvbench import (
    load_mvbench_tasks,
    format_mcq_prompt,
    extract_answer_mvbench,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-VL-7B-Instruct-8bit",
    )
    ap.add_argument(
        "--data-dir",
        default="/Volumes/RAID0/datasets/MVBench",
    )
    ap.add_argument("--max-tokens", type=int, default=30)
    ap.add_argument(
        "--output-dir", default="exp_text_only_output",
    )
    ap.add_argument(
        "--max-questions", type=int, default=260,
    )
    ap.add_argument(
        "--tasks", type=str, nargs="+", default=None,
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────
    print(f"Loading MVBench from {args.data_dir}...")
    all_tasks = load_mvbench_tasks(args.data_dir)
    if args.tasks:
        all_tasks = {
            k: v for k, v in all_tasks.items()
            if k in args.tasks
        }

    rng = np.random.RandomState(args.seed)
    sampled = {}
    per_task = max(
        1, args.max_questions // max(len(all_tasks), 1),
    )
    for tname, qs in all_tasks.items():
        n = min(per_task, len(qs))
        idx = rng.choice(len(qs), n, replace=False)
        sampled[tname] = [qs[i] for i in idx]
    total_q = sum(len(v) for v in sampled.values())
    print(f"  {total_q} questions across "
          f"{len(sampled)} tasks")

    # ── Load model ────────────────────────────────────
    from mlx_vlm import load
    from mlx_vlm.models.cache import make_prompt_cache

    print(f"\nLoading {args.model}...")
    model, processor = load(args.model)

    # ── Evaluate ──────────────────────────────────────
    results = []
    t0 = time.time()

    for tname in sorted(sampled.keys()):
        questions = sampled[tname]
        print(f"\n--- {tname} ({len(questions)} Qs) ---")

        for qi, item in enumerate(questions):
            question = item["question"]
            candidates = item["candidates"]
            gt = item["answer"]

            try:
                prompt = format_mcq_prompt(
                    question, candidates,
                )
                # Text-only: no video/image, just the question
                msgs = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }]
                text = processor.apply_chat_template(
                    msgs, tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = processor(
                    text=[text], padding=True,
                    return_tensors="pt",
                )
                ids = mx.array(inputs["input_ids"])
                attn_mask = mx.array(inputs["attention_mask"])

                # Simple prefill + generate
                cache = make_prompt_cache(
                    model.language_model,
                )
                embeds = model.language_model.model.embed_tokens(
                    ids,
                )
                h = model.language_model.model(
                    ids, cache=cache, inputs_embeds=embeds,
                )
                mx.eval([c.state for c in cache])
                if model.language_model.args.tie_word_embeddings:
                    logits = (
                        model.language_model.model
                        .embed_tokens.as_linear(h)
                    )
                else:
                    logits = model.language_model.lm_head(h)
                logits = logits[:, -1, :]

                # Generate
                tokens = []
                y = mx.argmax(logits, axis=-1)
                tokens.append(y.item())
                for _ in range(args.max_tokens - 1):
                    out = model.language_model(
                        y[None], cache=cache,
                    )
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

                out_text = processor.decode(
                    tokens, skip_special_tokens=True,
                )
                pred, _ = extract_answer_mvbench(
                    out_text, candidates,
                )
                correct = (pred == gt)

                result = {
                    "task": tname,
                    "qi": qi,
                    "gt": gt,
                    "pred": pred,
                    "correct": correct,
                    "output": out_text[:80],
                }
                results.append(result)

                mark = "O" if correct else "X"
                print(f"  Q{qi}: [{mark}] {pred[:30]}")

            except Exception as ex:
                print(f"  ERROR Q{qi}: {ex}")
                import traceback
                traceback.print_exc()

    # ── Summary ───────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"TEXT-ONLY BASELINE")
    print(f"  {len(results)} questions in "
          f"{elapsed/60:.1f} min")
    print(f"{'='*60}")

    if results:
        from collections import defaultdict
        tasks_r = defaultdict(list)
        for r in results:
            tasks_r[r['task']].append(r)

        print(f"\n{'Task':<25} {'N':>3} {'Acc':>6}")
        print('-' * 36)
        for t in sorted(tasks_r.keys()):
            rs = tasks_r[t]
            acc = sum(r['correct'] for r in rs) / len(rs)
            print(f"{t:<25} {len(rs):3d} {acc:5.1%}")

        total_acc = (
            sum(r['correct'] for r in results) / len(results)
        )
        print(f"\n{'OVERALL':<25} {len(results):3d} "
              f"{total_acc:5.1%}")

    # Save
    jp = os.path.join(args.output_dir, "results.json")
    with open(jp, "w") as f:
        json.dump({
            "experiment": "text_only_baseline",
            "seed": args.seed,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved → {jp}")


if __name__ == "__main__":
    main()
