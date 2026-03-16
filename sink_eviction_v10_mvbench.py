"""
v10: MVBench evaluation of streaming KV cache eviction.

Uses MVBench (4000 MCQ, 20 tasks) to validate v9's finding
that sink-aware streaming eviction preserves accuracy under
extreme compression.

MVBench format:
  {"video": "filename.webm", "question": "...",
   "candidates": ["A","B","C"], "answer": "correct answer"}

Protocol:
  For each question:
  1. Load video, process with Qwen2.5-VL processor
  2. Run streaming prefill with chunk-by-chunk eviction
  3. Generate answer, extract best candidate
  4. Compare against ground truth

Usage:
  uv run --with "transformers<4.52" --with torch \
    --with matplotlib \
    python sink_eviction_v10_mvbench.py \
      --data-dir /Volumes/RAID0/datasets/MVBench \
      --output-dir sink_eviction_output_v10 \
      --max-questions 200
"""

import argparse
import glob
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from sink_detect import (
    CaptureStore,
    find_image_token_range,
    patch_model,
    rmsnorm,
)

# Import streaming eviction from v9
from sink_eviction_v9_streaming import (
    streaming_prefill,
    generate_from_cache,
    full_prefill_generate,
)


# ── MVBench data loading ───────────────────────────────

def load_mvbench_tasks(data_dir: str) -> Dict[str, List[dict]]:
    """
    Load MVBench JSON task files.
    Returns {task_name: [question_dicts]}.
    """
    json_dir = os.path.join(data_dir, "json")
    if not os.path.isdir(json_dir):
        # Try flat structure
        json_files = glob.glob(
            os.path.join(data_dir, "*.json")
        )
        if not json_files:
            raise FileNotFoundError(
                f"No JSON files found in {data_dir} or "
                f"{json_dir}"
            )
    else:
        json_files = glob.glob(
            os.path.join(json_dir, "*.json")
        )

    tasks = {}
    for jf in sorted(json_files):
        task_name = Path(jf).stem
        with open(jf) as f:
            data = json.load(f)
        if isinstance(data, list):
            tasks[task_name] = data
        elif isinstance(data, dict) and "data" in data:
            tasks[task_name] = data["data"]
        else:
            print(f"  SKIP {task_name}: unexpected format")
    return tasks


_video_index = None


def _build_video_index(data_dir: str) -> Dict[str, str]:
    """Build filename→path index for all videos."""
    global _video_index
    if _video_index is not None:
        return _video_index

    _video_index = {}
    video_dir = os.path.join(data_dir, "video")
    if not os.path.isdir(video_dir):
        video_dir = data_dir

    for root, dirs, files in os.walk(video_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in (".mp4", ".webm", ".avi", ".mkv"):
                _video_index[f] = os.path.join(root, f)

    print(f"  Video index: {len(_video_index)} files")
    return _video_index


def find_video(
    video_name: str, data_dir: str,
    video_dirs: List[str] = None,
) -> Optional[str]:
    """Find video file using pre-built index."""
    idx = _build_video_index(data_dir)
    # Direct filename match
    if video_name in idx:
        return idx[video_name]
    # Try basename only (strip subdirs)
    basename = os.path.basename(video_name)
    if basename in idx:
        return idx[basename]
    return None


def format_mcq_prompt(question: str, candidates: List[str]):
    """Format MVBench question as MCQ prompt."""
    # MVBench uses candidates as plain strings
    letters = ["A", "B", "C", "D", "E"]
    choices = []
    for i, c in enumerate(candidates):
        if i < len(letters):
            choices.append(f"{letters[i]}. {c}")
    choice_text = "\n".join(choices)
    return (
        f"{question}\n{choice_text}\n"
        f"Answer with the letter only "
        f"({', '.join(letters[:len(candidates)])})."
    )


def extract_answer_mvbench(
    text: str, candidates: List[str],
) -> Tuple[str, int]:
    """
    Extract answer from model output.
    Returns (answer_text, candidate_index).
    """
    text = text.strip()
    letters = ["A", "B", "C", "D", "E"]

    # Try direct letter match
    m = re.search(r'\b([A-E])\b', text)
    if m:
        letter = m.group(1)
        idx = letters.index(letter)
        if idx < len(candidates):
            return candidates[idx], idx

    # Try matching candidate text directly
    text_lower = text.lower()
    for i, c in enumerate(candidates):
        if c.lower() in text_lower:
            return c, i

    # Fallback: first letter
    if text and text[0].upper() in letters:
        idx = letters.index(text[0].upper())
        if idx < len(candidates):
            return candidates[idx], idx

    return text[:50], -1


# ── Main evaluation ────────────────────────────────────

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
        "--output-dir", default="sink_eviction_output_v10",
    )
    ap.add_argument(
        "--n-chunks", type=int, default=5,
        help="Visual token chunks for streaming eviction",
    )
    ap.add_argument(
        "--max-questions", type=int, default=0,
        help="Max questions total (0=all). "
             "Sampled evenly across tasks.",
    )
    ap.add_argument(
        "--budgets", type=float, nargs="+",
        default=[0.10, 0.20, 0.40, 0.60],
    )
    ap.add_argument(
        "--strategies", type=str, nargs="+",
        default=["sink_aware", "anti_sink", "uniform",
                 "random"],
    )
    ap.add_argument(
        "--tasks", type=str, nargs="+", default=None,
        help="Specific task names to evaluate (default=all)",
    )
    ap.add_argument(
        "--resume", type=str, default=None,
        help="Resume from partial results JSON",
    )
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sink_dims = [458, 2570]
    budgets = args.budgets
    strategies = args.strategies

    # ── Load MVBench data ───────────────────────────────
    print(f"Loading MVBench from {args.data_dir}...")
    all_tasks = load_mvbench_tasks(args.data_dir)
    print(f"  Found {len(all_tasks)} tasks: "
          f"{', '.join(sorted(all_tasks.keys())[:5])}...")

    # Filter tasks if specified
    if args.tasks:
        all_tasks = {
            k: v for k, v in all_tasks.items()
            if k in args.tasks
        }

    # Count total questions and sample if needed
    total_q = sum(len(v) for v in all_tasks.values())
    print(f"  Total questions: {total_q}")

    if args.max_questions > 0 and args.max_questions < total_q:
        # Sample evenly across tasks
        per_task = max(
            1, args.max_questions // len(all_tasks),
        )
        sampled_tasks = {}
        rng = np.random.RandomState(42)
        for tname, questions in all_tasks.items():
            n = min(per_task, len(questions))
            indices = rng.choice(
                len(questions), n, replace=False,
            )
            sampled_tasks[tname] = [questions[i] for i in indices]
        all_tasks = sampled_tasks
        total_q = sum(len(v) for v in all_tasks.values())
        print(f"  Sampled to {total_q} questions "
              f"({per_task}/task)")

    # ── Load model ──────────────────────────────────────
    from mlx_vlm import load
    from mlx_vlm.video_generate import process_vision_info

    print(f"\nLoading {args.model}...")
    model, processor = load(args.model)
    patch_model(model)
    mcfg = model.config

    # ── Resume support ──────────────────────────────────
    completed = set()
    all_details = []
    if args.resume and os.path.exists(args.resume):
        with open(args.resume) as f:
            prev = json.load(f)
        all_details = prev.get("details", [])
        for d in all_details:
            key = (d["task"], d["qi"], d["strategy"],
                   d["budget"])
            completed.add(key)
        print(f"  Resuming: {len(completed)} results loaded")

    def prepare_video_input(video_path, question, candidates):
        """Prepare model input for a video MCQ question."""
        prompt = format_mcq_prompt(question, candidates)
        msgs = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
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

    # ── Evaluate ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"v10: MVBench STREAMING EVICTION EVALUATION")
    print(f"  {total_q} questions, "
          f"{len(budgets)} budgets, "
          f"{len(strategies)} strategies")
    print(f"  {args.n_chunks} visual chunks")
    print(f"{'='*60}")

    task_results = {}  # {task: {strategy: {budget: [bool]}}}
    n_done = 0
    n_skip = 0
    n_error = 0
    t0 = time.time()

    for tname in sorted(all_tasks.keys()):
        questions = all_tasks[tname]
        print(f"\n--- {tname} ({len(questions)} questions) ---")
        task_results[tname] = {
            sn: {b: [] for b in budgets}
            for sn in strategies + ["baseline"]
        }

        for qi, item in enumerate(questions):
            video_name = item["video"]
            question = item["question"]
            candidates = item["candidates"]
            gt_answer = item["answer"]

            # Find video file
            video_path = find_video(video_name, args.data_dir)
            if video_path is None:
                n_skip += 1
                if n_skip <= 3:
                    print(f"  SKIP Q{qi}: {video_name} "
                          f"not found")
                continue

            try:
                ids, pv, mask, extra = prepare_video_input(
                    video_path, question, candidates,
                )
                s, e = find_image_token_range(ids, mcfg)
                nv = e - s
            except Exception as ex:
                n_error += 1
                if n_error <= 5:
                    print(f"  ERROR Q{qi}: {ex}")
                continue

            # Baseline (no eviction)
            bl_key = (tname, qi, "baseline", 1.0)
            if bl_key not in completed:
                try:
                    out = full_prefill_generate(
                        model, processor, ids, pv, mask,
                        extra, max_tok=args.max_tokens,
                    )
                    pred_text, pred_idx = extract_answer_mvbench(
                        out, candidates,
                    )
                    correct = (pred_text == gt_answer)
                    task_results[tname]["baseline"][
                        budgets[0]
                    ].append(correct)
                    all_details.append({
                        "task": tname, "qi": qi,
                        "strategy": "baseline",
                        "budget": 1.0,
                        "pred": pred_text,
                        "gt": gt_answer,
                        "correct": correct,
                        "output": out[:80],
                    })
                except Exception as ex:
                    n_error += 1
                    if n_error <= 5:
                        print(f"  BL ERROR Q{qi}: {ex}")
                    continue

            # Streaming eviction strategies
            for budget in budgets:
                for sn in strategies:
                    key = (tname, qi, sn, budget)
                    if key in completed:
                        continue

                    try:
                        cache, logits, vcp = streaming_prefill(
                            model, processor, ids, pv, mask,
                            extra, s, e, budget, sn,
                            sink_dims,
                            n_chunks=args.n_chunks,
                        )
                        out = generate_from_cache(
                            model, processor, cache, logits,
                            max_tok=args.max_tokens,
                        )
                        pred_text, pred_idx = (
                            extract_answer_mvbench(
                                out, candidates,
                            )
                        )
                        correct = (pred_text == gt_answer)
                        task_results[tname][sn][budget].append(
                            correct,
                        )
                        all_details.append({
                            "task": tname, "qi": qi,
                            "strategy": sn,
                            "budget": budget,
                            "pred": pred_text,
                            "gt": gt_answer,
                            "correct": correct,
                            "output": out[:80],
                            "n_vis": nv,
                            "n_kept": len(vcp),
                        })
                    except Exception as ex:
                        n_error += 1
                        if n_error <= 10:
                            print(f"  ERR {sn} b={budget} "
                                  f"Q{qi}: {ex}")

            n_done += 1
            if n_done % 10 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed * 60
                print(f"  [{n_done}/{total_q}] "
                      f"{rate:.1f} q/min, "
                      f"{n_skip} skipped, "
                      f"{n_error} errors")

                # Checkpoint
                _save_results(
                    args.output_dir, all_details, budgets,
                    strategies, task_results, args,
                    checkpoint=True,
                )

    # ── Final results ───────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"COMPLETED: {n_done} questions in "
          f"{elapsed/60:.1f} min")
    print(f"  Skipped: {n_skip}, Errors: {n_error}")
    print(f"{'='*60}")

    _print_results(task_results, budgets, strategies)
    _save_results(
        args.output_dir, all_details, budgets,
        strategies, task_results, args,
    )


def _print_results(task_results, budgets, strategies):
    """Print aggregate accuracy table."""
    # Aggregate across tasks
    agg = {
        sn: {b: [] for b in budgets}
        for sn in strategies + ["baseline"]
    }
    for tname, tr in task_results.items():
        for sn in strategies + ["baseline"]:
            for b in budgets:
                agg[sn][b].extend(tr.get(sn, {}).get(b, []))

    # Baseline
    bl_items = agg["baseline"][budgets[0]]
    bl_acc = (sum(bl_items) / len(bl_items)) if bl_items else 0
    print(f"\nBaseline (no eviction): {bl_acc:.1%} "
          f"({sum(bl_items)}/{len(bl_items)})")

    # Table
    print(f"\n{'Budget':<8}", end="")
    for sn in strategies:
        print(f"{sn:<14}", end="")
    print()
    print("-" * (8 + 14 * len(strategies)))

    for b in budgets:
        print(f"{b:<8.0%}", end="")
        for sn in strategies:
            items = agg[sn][b]
            if items:
                acc = sum(items) / len(items)
                print(f"{acc:<14.1%}", end="")
            else:
                print(f"{'N/A':<14}", end="")
        print()

    # Per-task breakdown at 20% budget
    if 0.20 in budgets:
        print(f"\n--- Per-task accuracy at 20% budget ---")
        print(f"{'Task':<25}", end="")
        for sn in strategies:
            print(f"{sn:<14}", end="")
        print(f"{'baseline':<14}")
        print("-" * (25 + 14 * (len(strategies) + 1)))
        for tname in sorted(task_results.keys()):
            tr = task_results[tname]
            print(f"{tname[:24]:<25}", end="")
            for sn in strategies:
                items = tr.get(sn, {}).get(0.20, [])
                if items:
                    acc = sum(items) / len(items)
                    print(f"{acc:<14.1%}", end="")
                else:
                    print(f"{'N/A':<14}", end="")
            bl = tr.get("baseline", {}).get(budgets[0], [])
            if bl:
                acc = sum(bl) / len(bl)
                print(f"{acc:<14.1%}", end="")
            print()


def _save_results(
    output_dir, details, budgets, strategies,
    task_results, args, checkpoint=False,
):
    """Save results to JSON and plot."""
    suffix = "_checkpoint" if checkpoint else ""
    jp = os.path.join(output_dir, f"results{suffix}.json")
    with open(jp, "w") as f:
        json.dump({
            "experiment": "v10_mvbench_streaming",
            "n_chunks": args.n_chunks,
            "budgets": budgets,
            "strategies": strategies,
            "details": details,
        }, f, indent=2)

    if not checkpoint:
        print(f"JSON → {jp}")
        _plot_results(
            output_dir, details, budgets, strategies,
        )


def _plot_results(output_dir, details, budgets, strategies):
    """Plot accuracy vs budget."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "sink_aware": "#E91E63",
        "anti_sink": "#2196F3",
        "uniform": "#FF9800",
        "recency": "#4CAF50",
        "random": "#9E9E9E",
    }

    # Compute accuracy per strategy per budget
    acc = {}
    for sn in strategies:
        acc[sn] = {}
        for b in budgets:
            items = [
                d["correct"] for d in details
                if d["strategy"] == sn and d["budget"] == b
            ]
            if items:
                acc[sn][b] = sum(items) / len(items)

    # Baseline
    bl_items = [
        d["correct"] for d in details
        if d["strategy"] == "baseline"
    ]
    bl_acc = sum(bl_items) / len(bl_items) if bl_items else 0

    fig, ax = plt.subplots(figsize=(10, 6))
    for sn in strategies:
        if sn in acc and acc[sn]:
            bs = sorted(acc[sn].keys())
            vs = [acc[sn][b] for b in bs]
            lw = 2.5 if sn in ("sink_aware", "anti_sink") else 1.5
            ax.plot(
                bs, vs, "-o",
                color=colors.get(sn, "#666"),
                label=sn, linewidth=lw, markersize=6,
            )

    ax.axhline(y=bl_acc, color="black", linestyle="--",
               alpha=0.5, label=f"baseline ({bl_acc:.1%})")
    ax.set_xlabel("Budget (fraction of visual tokens kept)")
    ax.set_ylabel("MCQ Accuracy")
    ax.set_title(
        "v10: MVBench — Streaming KV Cache Eviction\n"
        f"{len(bl_items)} questions, ground-truth MCQ"
    )
    ax.legend()
    ax.set_xlim(0.05, 0.65)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fp = os.path.join(output_dir, "v10_mvbench.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot → {fp}")


if __name__ == "__main__":
    main()
