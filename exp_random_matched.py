"""
SM-random-matched experiment: mask 90% of visual tokens randomly.

Resolves the masking ratio confound from Gemini review:
  SM-sink masks ~90% of tokens on image benchmarks.
  Does ANY 90% masking cause the same drop, or is it
  specific to removing sinks?

Conditions per question:
  BL  — baseline (normal forward)
  R90 — SM-random-90% (mask 90% of visual tokens randomly)
  SM  — SM-sink (mask φ>τ tokens, for comparison)

Only runs on image benchmarks (POPE, ScienceQA, MMStar) where
sink fraction is 89-97% and the confound is strongest.

Usage:
  TOKENIZERS_PARALLELISM=false uv run \
    --with "transformers<4.52" --with torch \
    --with datasets --with matplotlib \
    python exp_random_matched.py \
      --benchmarks pope scienceqa mmstar \
      --max-per-benchmark 100 \
      --seed 42 \
      --output-dir exp_random_matched_seed42
"""

import argparse
import json
import os
import time
from typing import List

import mlx.core as mx
import numpy as np

from benchmark_loaders import (
    load_mmstar, load_pope, load_scienceqa_img,
    extract_answer, check_correct,
)
from exp_three_way_masking import (
    patch_model_v2, prepare_vision_input,
    run_baseline, run_soft_mask,
    find_image_token_range, SoftMask,
)
from sink_detect import CaptureStore, rmsnorm


# ── Structured JSONL logger ───────────────────────────

class ExpLogger:
    def __init__(self, path: str):
        self.f = open(path, "a")

    def log(self, event: str, **kw):
        kw["event"] = event
        kw["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        kw["cpu_load"] = round(os.getloadavg()[0], 2)
        try:
            kw["metal_mem_mb"] = round(
                mx.metal.get_active_memory() / 1e6, 1,
            )
            kw["metal_peak_mb"] = round(
                mx.metal.get_peak_memory() / 1e6, 1,
            )
        except Exception:
            pass
        self.f.write(json.dumps(kw, default=str) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def detect_sinks(
    hidden_states, img_start, img_end,
    sink_dims, layer=14, tau=20.0,
):
    """Return absolute sink indices in the full sequence."""
    hs = hidden_states[layer]
    if isinstance(hs, mx.array):
        hs = np.array(hs)
    if hs.ndim == 3:
        hs = hs[0]
    rms_val = np.abs(rmsnorm(hs))
    phi = np.max(
        np.stack(
            [rms_val[:, d] for d in sink_dims], axis=-1,
        ),
        axis=-1,
    )
    vis_phi = phi[img_start:img_end]
    sink_local = np.where(vis_phi > tau)[0]
    sink_abs = sink_local + img_start
    return sink_abs, vis_phi


def load_benchmark(name, max_samples, seed):
    if name == "mmstar":
        return load_mmstar(max_samples, seed)
    elif name == "pope":
        return load_pope(max_samples, seed)
    elif name == "scienceqa":
        return load_scienceqa_img(max_samples, seed)
    else:
        raise ValueError(f"Unknown benchmark: {name}")


def main():
    ap = argparse.ArgumentParser(
        description="SM-random-matched (90%) experiment",
    )
    ap.add_argument(
        "--model",
        default=(
            "mlx-community/Qwen2.5-VL-7B-Instruct-8bit"
        ),
    )
    ap.add_argument(
        "--benchmarks", nargs="+",
        default=["pope", "scienceqa", "mmstar"],
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
        default="exp_random_matched_output",
    )
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sink_dims = [458, 2570]
    rng = np.random.RandomState(args.seed)

    # ── Init logger ────────────────────────────────────
    log_path = os.path.join(
        args.output_dir, "exp_run.jsonl",
    )
    lg = ExpLogger(log_path)
    lg.log(
        "exp_start",
        experiment="random_matched_90pct",
        model=args.model, seed=args.seed,
        benchmarks=args.benchmarks,
        max_per_benchmark=args.max_per_benchmark,
        tau=args.tau, detect_layer=args.detect_layer,
        sink_dims=sink_dims,
    )

    # ── Load model ─────────────────────────────────────
    from mlx_vlm import load

    print(f"Loading {args.model}...")
    model, processor = load(args.model)
    patch_model_v2(model)
    mcfg = model.config

    # ── Run benchmarks ─────────────────────────────────
    all_results = {}
    t0_global = time.time()

    for bname in args.benchmarks:
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {bname}")
        print(f"{'='*60}")

        lg.log("bench_start", benchmark=bname)
        items = load_benchmark(
            bname, args.max_per_benchmark, args.seed,
        )
        print(f"  Loaded {len(items)} items")
        lg.log(
            "bench_loaded",
            benchmark=bname, n_items=len(items),
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

                # Resolve lazy image
                image = item["image"]
                if image is None and "_load_image" in item:
                    image = item["_load_image"]()

                # Prepare input
                ids, pv, attn_mask, extra = (
                    prepare_vision_input(
                        processor, question, candidates,
                        image=image,
                    )
                )
                s, e = find_image_token_range(ids, mcfg)
                seq_len = ids.shape[1]
                n_vis = e - s

                # ── BL: Baseline ───────────────────────
                (out_bl, hs_np, embeds, pos_ids,
                 rope_deltas, fkw) = run_baseline(
                    model, processor, ids, pv,
                    attn_mask, extra, args.max_tokens,
                )

                # ── Detect sinks ───────────────────────
                sink_abs, vis_phi = detect_sinks(
                    hs_np, s, e, sink_dims,
                    args.detect_layer, args.tau,
                )
                n_sink = len(sink_abs)
                sink_frac = (
                    round(n_sink / n_vis, 3)
                    if n_vis > 0 else 0
                )

                # ── SM-sink ────────────────────────────
                if n_sink > 0:
                    out_sm = run_soft_mask(
                        model, processor, ids, pv,
                        attn_mask, extra,
                        sink_abs, seq_len,
                        args.max_tokens,
                    )
                else:
                    out_sm = out_bl

                # ── SM-random-90% ──────────────────────
                n_mask_90 = int(0.9 * n_vis)
                if n_mask_90 > 0:
                    vis_indices = np.arange(s, e)
                    rand_90 = rng.choice(
                        vis_indices, size=n_mask_90,
                        replace=False,
                    )
                    out_r90 = run_soft_mask(
                        model, processor, ids, pv,
                        attn_mask, extra,
                        rand_90, seq_len,
                        args.max_tokens,
                    )
                else:
                    out_r90 = out_bl

                # ── Extract answers ────────────────────
                pred_bl = extract_answer(
                    bm, out_bl, candidates,
                )
                pred_sm = extract_answer(
                    bm, out_sm, candidates,
                )
                pred_r90 = extract_answer(
                    bm, out_r90, candidates,
                )

                q_elapsed = time.time() - q_start

                result = {
                    "benchmark": bname,
                    "qi": qi,
                    "gt": gt,
                    "elapsed_sec": round(q_elapsed, 2),
                    "n_vis": n_vis,
                    "n_sink": n_sink,
                    "sink_frac": sink_frac,
                    "n_mask_r90": n_mask_90,
                    "pred_bl": pred_bl,
                    "pred_sm": pred_sm,
                    "pred_r90": pred_r90,
                    "out_bl": out_bl[:80],
                    "out_sm": out_sm[:80],
                    "out_r90": out_r90[:80],
                    "correct_bl": check_correct(
                        bm, pred_bl, gt,
                    ),
                    "correct_sm": check_correct(
                        bm, pred_sm, gt,
                    ),
                    "correct_r90": check_correct(
                        bm, pred_r90, gt,
                    ),
                }
                results.append(result)

                m_bl = "O" if result["correct_bl"] else "X"
                m_sm = "O" if result["correct_sm"] else "X"
                m_r90 = (
                    "O" if result["correct_r90"] else "X"
                )
                print(
                    f"  [{qi+1}/{len(items)}] "
                    f"BL[{m_bl}] SM[{m_sm}] R90[{m_r90}]  "
                    f"sink={n_sink}/{n_vis}  "
                    f"{q_elapsed:.1f}s"
                )
                lg.log(
                    "question",
                    benchmark=bname, qi=qi,
                    n_vis=n_vis, n_sink=n_sink,
                    sink_frac=sink_frac,
                    n_mask_r90=n_mask_90,
                    correct_bl=result["correct_bl"],
                    correct_sm=result["correct_sm"],
                    correct_r90=result["correct_r90"],
                    elapsed=q_elapsed,
                )

                # Cleanup
                del hs_np, embeds, fkw
                mx.metal.clear_cache()

            except Exception as ex:
                print(f"  ERROR Q{qi}: {ex}")
                lg.log(
                    "error", benchmark=bname,
                    qi=qi, msg=str(ex),
                )
                import traceback
                traceback.print_exc()

        # ── Benchmark summary ──────────────────────────
        bench_elapsed = time.time() - t0_bench
        all_results[bname] = results
        n = len(results)

        if n > 0:
            acc_bl = sum(
                r["correct_bl"] for r in results
            ) / n
            acc_sm = sum(
                r["correct_sm"] for r in results
            ) / n
            acc_r90 = sum(
                r["correct_r90"] for r in results
            ) / n
            avg_sf = np.mean(
                [r["sink_frac"] for r in results],
            )

            print(f"\n  --- {bname} ({n} Qs, "
                  f"{bench_elapsed/60:.1f} min) ---")
            print(f"    Baseline:    {acc_bl:.1%}")
            print(f"    SM-sink:     {acc_sm:.1%}")
            print(f"    SM-rand-90%: {acc_r90:.1%}")
            print(f"    Avg sink frac: {avg_sf:.1%}")

            lg.log(
                "bench_end", benchmark=bname,
                n=n,
                elapsed_min=round(bench_elapsed / 60, 1),
                acc_bl=round(acc_bl, 3),
                acc_sm=round(acc_sm, 3),
                acc_r90=round(acc_r90, 3),
                avg_sink_frac=round(avg_sf, 3),
            )

        # Checkpoint
        _save(args.output_dir, all_results, args,
              sink_dims, checkpoint=True)

    # ── Final summary ──────────────────────────────────
    total_elapsed = time.time() - t0_global
    print(f"\n{'='*60}")
    print("SM-RANDOM-MATCHED (90%) EXPERIMENT")
    print(f"{'='*60}")

    print(f"\n{'Benchmark':<16}{'BL':<8}{'SM-sink':<10}"
          f"{'R90':<8}{'N':<6}{'Sink%':<8}")
    print("-" * 56)
    for bname, results in all_results.items():
        n = len(results)
        if n == 0:
            continue
        acc_bl = sum(
            r["correct_bl"] for r in results
        ) / n
        acc_sm = sum(
            r["correct_sm"] for r in results
        ) / n
        acc_r90 = sum(
            r["correct_r90"] for r in results
        ) / n
        avg_sf = np.mean(
            [r["sink_frac"] for r in results],
        )
        print(
            f"{bname:<16}{acc_bl:<8.1%}{acc_sm:<10.1%}"
            f"{acc_r90:<8.1%}{n:<6}{avg_sf:<8.1%}"
        )

    print(f"\nTotal time: {total_elapsed/60:.1f} min")

    print("\nInterpretation guide:")
    print("  If R90 ≈ SM-sink → any 90% masking causes "
          "the same drop. Structure doesn't matter.")
    print("  If R90 < SM-sink → sinks are structurally "
          "special. Random 90% is MORE harmful.")
    print("  If R90 > SM-sink → sinks are MORE important "
          "than random tokens (confirms anchor theory).")

    _save(args.output_dir, all_results, args, sink_dims)
    lg.log(
        "exp_end",
        total_min=round(total_elapsed / 60, 1),
    )
    lg.close()


def _save(output_dir, all_results, args, sink_dims,
          checkpoint=False):
    suffix = "_checkpoint" if checkpoint else ""
    fp = os.path.join(
        output_dir, f"results{suffix}.json",
    )
    with open(fp, "w") as f:
        json.dump({
            "experiment": "random_matched_90pct",
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
