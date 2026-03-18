"""
Multi-benchmark ablation evaluation.

Validates sink token findings across 6 benchmarks:
  Image: MMStar, POPE, ScienceQA-IMG
  Video: MVBench-Temporal, MVBench-Spatial, MVBench-Semantic

6 ablation conditions per question:
  BL (baseline), SM (soft-mask sink), HE (hard evict),
  RN (SM-random), AN (SM-anti-sink), TO (text-only)

Usage:
  uv run --with "transformers<4.52" --with torch \
    --with datasets --with matplotlib \
    python exp_multi_benchmark.py \
      --benchmarks mmstar pope scienceqa \
                   mvbench-temporal mvbench-spatial \
                   mvbench-semantic \
      --mvbench-dir /Volumes/RAID0/datasets/MVBench \
      --max-per-benchmark 100 \
      --seed 42 \
      --output-dir exp_multi_benchmark_output
"""

import argparse
import json
import os
import time
from typing import List

import mlx.core as mx
import numpy as np


# ── Structured JSONL logger ───────────────────────────

class ExpLogger:
    """Append-only JSONL logger for experiment tracking.

    Each line is a self-contained JSON object with timestamp.
    Analyze with: jq, pandas.read_json(path, lines=True),
    or grep.
    """

    def __init__(self, path: str):
        self.f = open(path, "a")

    def log(self, event: str, **kw):
        kw["event"] = event
        kw["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.f.write(json.dumps(kw, default=str) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


_lg: ExpLogger = None  # module-level, set in main()

from benchmark_loaders import (
    MVBENCH_TEMPORAL,
    MVBENCH_SPATIAL,
    MVBENCH_SEMANTIC,
    load_mmstar,
    load_pope,
    load_scienceqa_img,
    load_mvbench_slice,
    extract_answer,
    check_correct,
)
from exp_three_way_masking import (
    patch_model_v2,
    prepare_vision_input,
    run_all_ablations,
    run_baseline,
    find_image_token_range,
)
from sink_detect import discover_sink_dims, CaptureStore


ALL_BENCHMARKS = [
    "mmstar", "pope", "scienceqa",
    "mvbench-temporal", "mvbench-spatial",
    "mvbench-semantic",
]


def load_benchmark(
    name: str, max_samples: int, seed: int,
    mvbench_dir: str = "",
) -> List[dict]:
    """Load items for a named benchmark."""
    if name == "mmstar":
        return load_mmstar(max_samples, seed)
    elif name == "pope":
        return load_pope(max_samples, seed)
    elif name == "scienceqa":
        return load_scienceqa_img(max_samples, seed)
    elif name == "mvbench-temporal":
        per_task = max(
            1, max_samples // len(MVBENCH_TEMPORAL),
        )
        return load_mvbench_slice(
            mvbench_dir, MVBENCH_TEMPORAL,
            per_task, seed,
        )
    elif name == "mvbench-spatial":
        per_task = max(
            1, max_samples // len(MVBENCH_SPATIAL),
        )
        return load_mvbench_slice(
            mvbench_dir, MVBENCH_SPATIAL,
            per_task, seed,
        )
    elif name == "mvbench-semantic":
        per_task = max(
            1, max_samples // len(MVBENCH_SEMANTIC),
        )
        return load_mvbench_slice(
            mvbench_dir, MVBENCH_SEMANTIC,
            per_task, seed,
        )
    else:
        raise ValueError(f"Unknown benchmark: {name}")


def run_text_only(
    model, processor, question, mcfg, max_tokens=30,
):
    """Run baseline with no visual input (text-only)."""
    from sink_eviction_v9_streaming import generate_from_cache
    from mlx_vlm.models.cache import make_prompt_cache
    from mlx_vlm.video_generate import process_vision_info

    msgs = [{
        "role": "user",
        "content": [{"type": "text", "text": question}],
    }]
    text = processor.apply_chat_template(
        msgs, tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text], images=None,
        videos=None, padding=True,
        return_tensors="pt",
    )
    ids = mx.array(inputs["input_ids"])

    CaptureStore.disable()
    from exp_three_way_masking import SoftMask
    SoftMask.disable()

    cache = make_prompt_cache(model.language_model)

    # Text-only: embed without pixel values
    model.language_model._position_ids = None
    model.language_model._rope_deltas = None

    embed_tokens = model.language_model.model.embed_tokens
    embeds = embed_tokens(ids)

    out = model.language_model(
        ids, inputs_embeds=embeds, cache=cache,
    )
    mx.eval([c.state for c in cache])
    logits = out.logits[:, -1, :]

    result = generate_from_cache(
        model, processor, cache, logits, max_tokens,
    )
    del cache
    mx.metal.clear_cache()
    return result


def main():
    ap = argparse.ArgumentParser(
        description="Multi-benchmark ablation evaluation",
    )
    ap.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-VL-7B-Instruct-8bit",
    )
    ap.add_argument(
        "--benchmarks", nargs="+",
        default=ALL_BENCHMARKS,
        choices=ALL_BENCHMARKS,
    )
    ap.add_argument(
        "--mvbench-dir",
        default="/Volumes/RAID0/datasets/MVBench",
    )
    ap.add_argument(
        "--max-per-benchmark", type=int, default=100,
    )
    ap.add_argument("--max-tokens", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tau", type=float, default=20.0)
    ap.add_argument("--detect-layer", type=int, default=14)
    ap.add_argument(
        "--output-dir",
        default="exp_multi_benchmark_output",
    )
    ap.add_argument(
        "--discover-dims", action="store_true",
        help="Auto-discover sink dims on first image Q",
    )
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sink_dims = [458, 2570]

    # ── Init JSONL logger ─────────────────────────────
    global _lg
    log_path = os.path.join(args.output_dir, "exp_run.jsonl")
    _lg = ExpLogger(log_path)
    _lg.log("exp_start",
            model=args.model, seed=args.seed,
            benchmarks=args.benchmarks,
            max_per_benchmark=args.max_per_benchmark,
            tau=args.tau, detect_layer=args.detect_layer,
            sink_dims=sink_dims)

    # ── Load model ────────────────────────────────────
    from mlx_vlm import load

    print(f"Loading {args.model}...")
    _lg.log("model_load_start", model=args.model)
    model, processor = load(args.model)
    patch_model_v2(model)
    mcfg = model.config
    rng = np.random.RandomState(args.seed)
    _lg.log("model_load_done")

    # ── Run benchmarks ────────────────────────────────
    all_results = {}
    t0_global = time.time()
    dims_discovered_for_image = False

    for bname in args.benchmarks:
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {bname}")
        print(f"{'='*60}")

        _lg.log("bench_start", benchmark=bname)
        items = load_benchmark(
            bname, args.max_per_benchmark,
            args.seed, args.mvbench_dir,
        )
        print(f"  Loaded {len(items)} items")
        _lg.log("bench_loaded", benchmark=bname,
                n_items=len(items))

        if not items:
            print("  SKIP: no items loaded")
            continue

        # Auto-discover sink dims for image benchmarks
        has_loader = "_load_image" in items[0]
        is_image = (
            items[0]["image"] is not None or has_loader
        )
        if (
            args.discover_dims
            and is_image
            and not dims_discovered_for_image
        ):
            print("  Discovering sink dims for images...")
            item0 = items[0]
            img0 = (
                item0["_load_image"]()
                if has_loader else item0["image"]
            )
            ids0, pv0, mask0, extra0 = prepare_vision_input(
                processor, item0["question"],
                item0["candidates"],
                image=img0,
            )
            del img0
            # Run baseline to get hidden states
            CaptureStore.enable()
            from exp_three_way_masking import SoftMask
            SoftMask.disable()
            from mlx_vlm.models.cache import make_prompt_cache
            cache = make_prompt_cache(model.language_model)
            eo = model.get_input_embeddings(
                ids0, pv0, mask=mask0, **extra0,
            )
            embeds = eo.inputs_embeds
            fkw = {
                k: v for k, v in eo.to_dict().items()
                if k != "inputs_embeds" and v is not None
            }
            _ = model.language_model(
                ids0, inputs_embeds=embeds,
                cache=cache, **fkw,
            )
            hs = [
                np.array(h)
                for h in CaptureStore.hidden_states
            ]
            CaptureStore.disable()
            img_dims, _ = discover_sink_dims(hs)
            print(f"  Image sink dims: {img_dims}")
            if img_dims != sink_dims:
                print(
                    f"  WARNING: differs from video dims "
                    f"{sink_dims}! Using image dims."
                )
                sink_dims = img_dims
            dims_discovered_for_image = True
            del cache, hs, eo, embeds
            mx.metal.clear_cache()

        # Print sink dims being used
        s_e_info = "image" if is_image else "video"
        print(
            f"  sink_dims={sink_dims} "
            f"({s_e_info} modality)"
        )

        results = []
        t0_bench = time.time()

        for qi, item in enumerate(items):
            try:
                q_start = time.time()
                question = item["question"]
                candidates = item["candidates"]
                gt = item["gt"]
                bm = item["benchmark"]

                # ── Resolve lazy image if needed ──────
                image = item["image"]
                if image is None and "_load_image" in item:
                    image = item["_load_image"]()

                # ── Prepare vision input ──────────────
                ids, pv, attn_mask, extra = (
                    prepare_vision_input(
                        processor, question, candidates,
                        image=image,
                        video_path=item["video_path"],
                    )
                )

                s, e = find_image_token_range(ids, mcfg)

                # ── Run 5 ablations ───────────────────
                abl = run_all_ablations(
                    model, processor, ids, pv,
                    attn_mask, extra, mcfg, sink_dims,
                    args.tau, args.detect_layer,
                    rng, args.max_tokens,
                )

                # ── Text-only ─────────────────────────
                out_to = run_text_only(
                    model, processor, question,
                    mcfg, args.max_tokens,
                )

                # ── Extract answers + check ───────────
                pred_bl = extract_answer(
                    bm, abl["out_bl"], candidates,
                )
                pred_sm = extract_answer(
                    bm, abl["out_sm"], candidates,
                )
                pred_he = extract_answer(
                    bm, abl["out_he"], candidates,
                )
                pred_rand = extract_answer(
                    bm, abl["out_rand"], candidates,
                )
                pred_anti = extract_answer(
                    bm, abl["out_anti"], candidates,
                )
                pred_to = extract_answer(
                    bm, out_to, candidates,
                )

                q_elapsed = time.time() - q_start

                result = {
                    "benchmark": bname,
                    "qi": qi,
                    "gt": gt,
                    "elapsed_sec": round(q_elapsed, 2),
                    "n_vis": abl["n_vis"],
                    "n_sink": abl["n_sink"],
                    "sink_frac": abl["sink_frac"],
                    "img_start": abl["img_start"],
                    "img_end": abl["img_end"],
                    "pred_bl": pred_bl,
                    "pred_sm": pred_sm,
                    "pred_he": pred_he,
                    "pred_rand": pred_rand,
                    "pred_anti": pred_anti,
                    "pred_to": pred_to,
                    "out_bl": abl["out_bl"][:80],
                    "out_sm": abl["out_sm"][:80],
                    "out_he": abl["out_he"][:80],
                    "out_rand": abl["out_rand"][:80],
                    "out_anti": abl["out_anti"][:80],
                    "out_to": out_to[:80],
                    "correct_bl": check_correct(
                        bm, pred_bl, gt,
                    ),
                    "correct_sm": check_correct(
                        bm, pred_sm, gt,
                    ),
                    "correct_he": check_correct(
                        bm, pred_he, gt,
                    ),
                    "correct_rand": check_correct(
                        bm, pred_rand, gt,
                    ),
                    "correct_anti": check_correct(
                        bm, pred_anti, gt,
                    ),
                    "correct_to": check_correct(
                        bm, pred_to, gt,
                    ),
                    "metadata": item.get("metadata", {}),
                }
                results.append(result)

                # Print status
                marks = "".join([
                    "O" if result[f"correct_{c}"] else "X"
                    for c in [
                        "bl", "sm", "he", "rand",
                        "anti", "to",
                    ]
                ])
                print(
                    f"  [{qi+1}/{len(items)}] "
                    f"BL SM HE RN AN TO = {marks}  "
                    f"sink={abl['n_sink']}/{abl['n_vis']}  "
                    f"{q_elapsed:.1f}s"
                )
                _lg.log("question",
                        benchmark=bname, qi=qi,
                        n_vis=abl["n_vis"],
                        n_sink=abl["n_sink"],
                        sink_frac=abl["sink_frac"],
                        correct_bl=result["correct_bl"],
                        correct_sm=result["correct_sm"],
                        correct_he=result["correct_he"],
                        correct_rand=result["correct_rand"],
                        correct_anti=result["correct_anti"],
                        correct_to=result["correct_to"],
                        elapsed=q_elapsed)

            except Exception as ex:
                print(f"  ERROR Q{qi}: {ex}")
                _lg.log("error", benchmark=bname, qi=qi,
                        msg=str(ex))
                import traceback
                traceback.print_exc()

        # ── Benchmark summary ─────────────────────────
        bench_elapsed = time.time() - t0_bench
        all_results[bname] = results
        _print_benchmark_summary(bname, results, bench_elapsed)

        n = len(results)
        if n > 0:
            _lg.log("bench_end", benchmark=bname,
                    n=n, elapsed_min=round(bench_elapsed/60, 1),
                    acc_bl=round(sum(r["correct_bl"] for r in results)/n, 3),
                    acc_sm=round(sum(r["correct_sm"] for r in results)/n, 3),
                    acc_he=round(sum(r["correct_he"] for r in results)/n, 3),
                    acc_rand=round(sum(r["correct_rand"] for r in results)/n, 3),
                    acc_anti=round(sum(r["correct_anti"] for r in results)/n, 3),
                    acc_to=round(sum(r["correct_to"] for r in results)/n, 3))

        # Checkpoint after each benchmark
        _save_results(
            args.output_dir, all_results, args,
            sink_dims, checkpoint=True,
        )

    # ── Cross-benchmark comparison ────────────────────
    total_elapsed = time.time() - t0_global
    print(f"\n{'='*60}")
    print("CROSS-BENCHMARK COMPARISON")
    print(f"{'='*60}")
    _print_cross_benchmark(all_results)
    print(f"\nTotal time: {total_elapsed/60:.1f} min")

    _save_results(
        args.output_dir, all_results, args, sink_dims,
    )
    _lg.log("exp_end",
            total_min=round(total_elapsed/60, 1),
            n_benchmarks=len(all_results))
    _lg.close()


def _print_benchmark_summary(
    bname: str, results: list, elapsed: float,
):
    """Print accuracy summary for one benchmark."""
    n = len(results)
    if n == 0:
        print(f"\n  {bname}: no results")
        return

    conditions = ["bl", "sm", "he", "rand", "anti", "to"]
    labels = [
        "Baseline", "SM-sink", "Hard evict",
        "SM-random", "SM-anti", "Text-only",
    ]

    print(f"\n  --- {bname} ({n} Qs, "
          f"{elapsed/60:.1f} min) ---")
    for cond, label in zip(conditions, labels):
        acc = sum(
            r[f"correct_{cond}"] for r in results
        ) / n
        print(f"    {label:<14s}: {acc:.1%}")

    avg_sink = np.mean(
        [r["sink_frac"] for r in results],
    )
    print(f"    Avg sink frac : {avg_sink:.1%}")


def _print_cross_benchmark(all_results: dict):
    """Print comparison table across benchmarks."""
    conditions = ["bl", "sm", "he", "rand", "anti", "to"]
    labels = ["BL", "SM", "HE", "RN", "AN", "TO"]

    # Header
    print(f"\n{'Benchmark':<22}", end="")
    for lb in labels:
        print(f"{lb:<8}", end="")
    print(f"{'N':<6}{'Sink%':<8}")
    print("-" * (22 + 8 * len(labels) + 14))

    for bname, results in all_results.items():
        n = len(results)
        if n == 0:
            continue
        print(f"{bname:<22}", end="")
        for cond in conditions:
            acc = sum(
                r[f"correct_{cond}"] for r in results
            ) / n
            print(f"{acc:<8.1%}", end="")
        avg_sink = np.mean(
            [r["sink_frac"] for r in results],
        )
        print(f"{n:<6}{avg_sink:<8.1%}")


def _save_results(
    output_dir: str, all_results: dict,
    args, sink_dims: list, checkpoint: bool = False,
):
    """Save all results to JSON."""
    suffix = "_checkpoint" if checkpoint else ""
    fp = os.path.join(
        output_dir, f"results{suffix}.json",
    )
    with open(fp, "w") as f:
        json.dump({
            "experiment": "multi_benchmark_ablation",
            "seed": args.seed,
            "tau": args.tau,
            "detect_layer": args.detect_layer,
            "sink_dims": sink_dims,
            "max_per_benchmark": args.max_per_benchmark,
            "benchmarks": {
                bname: results
                for bname, results in all_results.items()
            },
        }, f, indent=2, default=str)
    if not checkpoint:
        print(f"\nResults saved -> {fp}")


if __name__ == "__main__":
    main()
