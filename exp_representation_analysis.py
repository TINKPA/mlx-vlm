"""
Representation analysis: how sink masking affects hidden states & logits.

Runs 3 conditions (BL, SM-sink, SM-random-90%) on image benchmarks,
captures all-layer hidden states + final logits, computes 6 metrics:
  1. Singular Value Entropy (SVE) per layer
  2. L2 norm per layer
  3. Cosine similarity (BL vs masked) per layer
  4. KL divergence (output logits)
  5. Prediction entropy (output logits)
  6. Top-K probability mass retention

Usage:
  TOKENIZERS_PARALLELISM=false uv run \
    --with "transformers<4.52" --with torch \
    --with datasets --with matplotlib --with scipy \
    python exp_representation_analysis.py \
      --benchmarks mmstar pope \
      --max-per-benchmark 50 \
      --seed 42 \
      --save-raw 5 \
      --output-dir ../../experiments/repr_analysis/full
"""

import argparse
import gc
import json
import os
import time
from typing import Dict, List, Tuple

import mlx.core as mx
import numpy as np


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


# ── Metric functions ─────────────────────────────────

def log_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax."""
    c = x.max()
    logsumexp = c + np.log(np.sum(np.exp(x - c)))
    return x - logsumexp


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    return np.exp(log_softmax(x))


def compute_sve(H: np.ndarray) -> float:
    """Singular Value Entropy for hidden state matrix H [N, D]."""
    try:
        s = np.linalg.svd(H, compute_uv=False)
    except np.linalg.LinAlgError:
        # SVD non-convergence: try with float64, then scipy
        try:
            s = np.linalg.svd(
                H.astype(np.float64), compute_uv=False,
            )
        except np.linalg.LinAlgError:
            from scipy.linalg import svd as scipy_svd
            try:
                s = scipy_svd(
                    H.astype(np.float64),
                    compute_uv=False, lapack_driver="gesdd",
                )
            except Exception:
                return float("nan")
    total = s.sum()
    if total < 1e-12:
        return 0.0
    p = s / total
    p = p[p > 1e-12]
    return float(-np.sum(p * np.log(p)))


def compute_l2_norm(H: np.ndarray) -> float:
    """Mean L2 norm across tokens. H: [N, D]."""
    return float(np.mean(np.linalg.norm(H, axis=1)))


def compute_cosine_sim(
    H_bl: np.ndarray, H_masked: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Per-token cosine similarity, return (mean, per_token)."""
    dot = np.sum(H_bl * H_masked, axis=1)
    norm_bl = np.linalg.norm(H_bl, axis=1)
    norm_m = np.linalg.norm(H_masked, axis=1)
    cos = dot / (norm_bl * norm_m + 1e-12)
    return float(np.mean(cos)), cos


def compute_kl_divergence(
    logits_bl: np.ndarray, logits_masked: np.ndarray,
) -> float:
    """KL(P_bl || P_masked) in bits."""
    log_p = log_softmax(logits_bl)
    log_q = log_softmax(logits_masked)
    p = np.exp(log_p)
    kl = float(np.sum(p * (log_p - log_q)) / np.log(2))
    return kl


def compute_entropy(logits: np.ndarray) -> float:
    """Prediction entropy in bits."""
    log_p = log_softmax(logits)
    p = np.exp(log_p)
    return float(-np.sum(p * log_p) / np.log(2))


def compute_topk_mass(
    logits_bl: np.ndarray,
    logits_masked: np.ndarray,
    ks: List[int] = None,
) -> Dict[str, float]:
    """Fraction of BL's top-k probability mass retained."""
    if ks is None:
        ks = [1, 5, 10]
    p_bl = softmax(logits_bl)
    p_masked = softmax(logits_masked)
    result = {}
    for k in ks:
        topk_idx = np.argsort(p_bl)[-k:]
        mass_bl = p_bl[topk_idx].sum()
        mass_masked = p_masked[topk_idx].sum()
        retention = float(mass_masked / mass_bl) if mass_bl > 0 else 0.0
        result[str(k)] = round(retention, 4)
    return result


def get_top1_token(logits: np.ndarray, tokenizer) -> str:
    """Decode top-1 predicted token."""
    idx = int(np.argmax(logits))
    return tokenizer.decode([idx]).strip()


# ── Forward pass with capture ─────────────────────────

def forward_with_capture(
    model, ids, pv, attn_mask, extra,
):
    """
    Single forward pass (prefill) capturing hidden states + logits.

    Returns (hidden_states: List[np.ndarray], logits: np.ndarray).
    hidden_states[i] has shape [seq_len, hidden_dim].
    logits has shape [vocab_size] (last token position).
    """
    from mlx_vlm.models.cache import make_prompt_cache
    from sink_detect import CaptureStore

    CaptureStore.enable()

    cache = make_prompt_cache(model.language_model)
    eo = model.get_input_embeddings(
        ids, pv, mask=attn_mask, **extra,
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

    # Extract logits at last position
    logits_mx = out.logits[0, -1, :]
    logits_np = np.array(logits_mx).astype(np.float32)

    # Convert hidden states to numpy
    # CaptureStore stores [B, L, D]; squeeze batch dim to [L, D]
    hs_np = []
    for h in CaptureStore.hidden_states:
        arr = np.array(h).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        hs_np.append(arr)

    CaptureStore.disable()
    del cache, eo, embeds, out
    mx.metal.clear_cache()

    return hs_np, logits_np


def forward_with_mask(
    model, ids, pv, attn_mask, extra,
    blocked_positions, seq_len,
):
    """
    Forward pass with soft mask (blocking attention to positions)
    AND hidden state capture.
    """
    from mlx_vlm.models.cache import make_prompt_cache
    from sink_detect import CaptureStore
    from exp_three_way_masking import SoftMask

    CaptureStore.enable()
    SoftMask.set_blocked(blocked_positions.tolist(), seq_len)

    model.language_model._position_ids = None
    model.language_model._rope_deltas = None

    cache = make_prompt_cache(model.language_model)
    eo = model.get_input_embeddings(
        ids, pv, mask=attn_mask, **extra,
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

    logits_mx = out.logits[0, -1, :]
    logits_np = np.array(logits_mx).astype(np.float32)

    hs_np = []
    for h in CaptureStore.hidden_states:
        arr = np.array(h).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        hs_np.append(arr)

    CaptureStore.disable()
    SoftMask.disable()
    del cache, eo, embeds, out
    mx.metal.clear_cache()

    return hs_np, logits_np


def forward_hard_evict(
    model, ids, pv, attn_mask, extra,
    evict_positions, seq_len,
):
    """
    Forward pass with tokens physically removed + capture.
    """
    from mlx_vlm.models.cache import make_prompt_cache
    from sink_detect import CaptureStore
    from exp_three_way_masking import SoftMask

    CaptureStore.enable()
    SoftMask.disable()

    model.language_model._position_ids = None
    model.language_model._rope_deltas = None

    cache = make_prompt_cache(model.language_model)
    eo = model.get_input_embeddings(
        ids, pv, mask=attn_mask, **extra,
    )
    embeds = eo.inputs_embeds
    pos_ids = model.language_model._position_ids
    rope_deltas = model.language_model._rope_deltas

    # Build keep mask
    keep = np.ones(embeds.shape[1], dtype=bool)
    for p in evict_positions:
        if p < embeds.shape[1]:
            keep[p] = False
    keep_idx = mx.array(np.where(keep)[0])

    evicted_embeds = embeds[:, keep_idx, :]
    evicted_ids = ids[:, keep_idx]
    evicted_len = evicted_embeds.shape[1]

    if pos_ids is not None:
        evicted_pos = pos_ids[:, :, keep_idx]
    else:
        evicted_pos = None

    model.language_model._position_ids = evicted_pos
    if evicted_pos is not None:
        max_pos = evicted_pos.max()
        model.language_model._rope_deltas = (
            max_pos + 1 - evicted_len
        )
    else:
        model.language_model._rope_deltas = rope_deltas

    h = model.language_model.model(
        evicted_ids, cache=cache,
        inputs_embeds=evicted_embeds,
        position_ids=evicted_pos,
    )
    mx.eval([c.state for c in cache])

    if model.language_model.args.tie_word_embeddings:
        logits = (
            model.language_model.model.embed_tokens
            .as_linear(h)
        )
    else:
        logits = model.language_model.lm_head(h)
    logits_np = np.array(logits[0, -1, :]).astype(np.float32)

    # HE has different seq_len — can't directly compare
    # hidden states per-token with BL. Store as-is.
    hs_np = []
    for h_layer in CaptureStore.hidden_states:
        arr = np.array(h_layer).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        hs_np.append(arr)

    CaptureStore.disable()
    del cache, eo, embeds, h
    mx.metal.clear_cache()

    return hs_np, logits_np


def forward_pixel_masked(
    model, ids, pv, attn_mask, extra,
    sink_local_indices, grid_thw, spatial_merge_size=2,
):
    """
    Forward pass with sink patches zeroed out in pixel_values.

    Tokens remain at same positions but carry "black" content.
    Tests whether attention still flows to these positions
    (position-driven) or shifts away (content-driven).
    """
    from exp_pixel_masking import sink_indices_to_pixel_mask

    T, H, W = grid_thw
    patch_idx = sink_indices_to_pixel_mask(
        sink_local_indices, (T, H, W), spatial_merge_size,
    )

    # Zero out the selected patches
    pv_masked = mx.array(np.array(pv))  # copy
    pv_np = np.array(pv_masked)
    valid = patch_idx[patch_idx < pv_np.shape[0]]
    if len(valid) > 0:
        pv_np[valid] = 0.0
    pv_masked = mx.array(pv_np)

    return forward_with_capture(
        model, ids, pv_masked, attn_mask, extra,
    )


def forward_text_only(
    model, processor, question,
):
    """
    Forward pass with no visual input + capture.
    """
    from mlx_vlm.models.cache import make_prompt_cache
    from sink_detect import CaptureStore
    from exp_three_way_masking import SoftMask
    from mlx_vlm.video_generate import process_vision_info

    CaptureStore.enable()
    SoftMask.disable()

    model.language_model._position_ids = None
    model.language_model._rope_deltas = None

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

    cache = make_prompt_cache(model.language_model)
    embed_tokens = model.language_model.model.embed_tokens
    embeds = embed_tokens(ids)

    out = model.language_model(
        ids, inputs_embeds=embeds, cache=cache,
    )
    mx.eval([c.state for c in cache])

    logits_np = np.array(
        out.logits[0, -1, :],
    ).astype(np.float32)

    hs_np = []
    for h in CaptureStore.hidden_states:
        arr = np.array(h).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        hs_np.append(arr)

    CaptureStore.disable()
    del cache, embeds, out
    mx.metal.clear_cache()

    return hs_np, logits_np


# ── Per-sample analysis ──────────────────────────────

# Condition labels used throughout
CONDS = ["bl", "sm", "he", "anti", "r50", "r90", "to"]


def _metrics_for_pair(hs_bl, hs_other, logits_bl, logits_other,
                      tokenizer, n_layers, same_len=True):
    """Compute per-layer + logit metrics for one BL-vs-X pair."""
    sve, norm, cosine = [], [], []
    for li in range(n_layers):
        sve.append(round(compute_sve(hs_other[li]), 4))
        norm.append(round(compute_l2_norm(hs_other[li]), 4))
        if same_len:
            m, _ = compute_cosine_sim(
                hs_bl[li], hs_other[li],
            )
            cosine.append(round(m, 6))
        else:
            cosine.append(float("nan"))

    kl = round(
        compute_kl_divergence(logits_bl, logits_other), 4,
    )
    entropy = round(compute_entropy(logits_other), 4)
    topk = compute_topk_mass(logits_bl, logits_other)
    top1 = get_top1_token(logits_other, tokenizer)

    return {
        "sve": sve, "norm": norm, "cosine": cosine,
        "kl": kl, "entropy": entropy,
        "topk": topk, "top1": top1,
    }


def analyze_sample(
    model, processor, ids, pv, attn_mask, extra,
    mcfg, sink_dims, tau, detect_layer, rng,
    question="",
):
    """
    Run 7 conditions, compute all metrics for one sample.

    Conditions: BL, SM-sink, HE, SM-anti-sink,
                SM-random-50%, SM-random-90%, Text-only
    """
    from sink_detect import find_image_token_range, rmsnorm

    s, e = find_image_token_range(ids, mcfg)
    seq_len = ids.shape[1]
    n_vis = e - s

    # ── BL ─────────────────────────────────────────────
    hs_bl, logits_bl = forward_with_capture(
        model, ids, pv, attn_mask, extra,
    )

    # Detect sinks
    hs_detect = hs_bl[detect_layer]
    rms_val = np.abs(rmsnorm(hs_detect))
    phi = np.max(
        np.stack([rms_val[:, d] for d in sink_dims], axis=-1),
        axis=-1,
    )
    vis_phi = phi[s:e]
    sink_local = np.where(vis_phi > tau)[0]
    sink_abs = sink_local + s
    n_sink = len(sink_abs)

    # ── SM-sink ────────────────────────────────────────
    if n_sink > 0:
        hs_sm, logits_sm = forward_with_mask(
            model, ids, pv, attn_mask, extra,
            sink_abs, seq_len,
        )
    else:
        hs_sm, logits_sm = hs_bl, logits_bl

    # ── HE (Hard Evict) ───────────────────────────────
    if n_sink > 0:
        hs_he, logits_he = forward_hard_evict(
            model, ids, pv, attn_mask, extra,
            sink_abs, seq_len,
        )
    else:
        hs_he, logits_he = hs_bl, logits_bl

    # ── PM (Pixel Masking) ──────────────────────────────
    # Zero out sink patches in pixel_values, re-run through
    # vision encoder. Same positions, "black" content.
    if n_sink > 0 and pv is not None:
        # Get grid_thw for spatial mapping
        grid_key = (
            "image_grid_thw"
            if "image_grid_thw" in extra
            else "video_grid_thw"
        )
        if grid_key in extra:
            grid_thw = tuple(
                int(x) for x in np.array(extra[grid_key][0])
            )
            sms = getattr(
                mcfg, "vision_config", mcfg,
            )
            sms_val = getattr(sms, "spatial_merge_size", 2)
            hs_pm, logits_pm = forward_pixel_masked(
                model, ids, pv, attn_mask, extra,
                sink_local, grid_thw, sms_val,
            )
        else:
            hs_pm, logits_pm = hs_bl, logits_bl
    else:
        hs_pm, logits_pm = hs_bl, logits_bl

    # ── SM-anti-sink ───────────────────────────────────
    if n_sink > 0 and n_sink < n_vis:
        all_vis = set(range(s, e))
        sink_set = set(sink_abs.tolist())
        anti_sink = np.array(sorted(all_vis - sink_set))
        hs_anti, logits_anti = forward_with_mask(
            model, ids, pv, attn_mask, extra,
            anti_sink, seq_len,
        )
    else:
        hs_anti, logits_anti = hs_bl, logits_bl

    # ── SM-random-50% ─────────────────────────────────
    vis_indices = np.arange(s, e)
    n_mask_50 = int(0.5 * n_vis)
    if n_mask_50 > 0:
        rand_50 = rng.choice(
            vis_indices, size=n_mask_50, replace=False,
        )
        hs_r50, logits_r50 = forward_with_mask(
            model, ids, pv, attn_mask, extra,
            rand_50, seq_len,
        )
    else:
        hs_r50, logits_r50 = hs_bl, logits_bl

    # ── SM-random-90% ─────────────────────────────────
    n_mask_90 = int(0.9 * n_vis)
    if n_mask_90 > 0:
        rand_90 = rng.choice(
            vis_indices, size=n_mask_90, replace=False,
        )
        hs_r90, logits_r90 = forward_with_mask(
            model, ids, pv, attn_mask, extra,
            rand_90, seq_len,
        )
    else:
        hs_r90, logits_r90 = hs_bl, logits_bl

    # ── TO (Text-only) ────────────────────────────────
    hs_to, logits_to = forward_text_only(
        model, processor, question,
    )

    # ── Compute metrics ────────────────────────────────
    n_layers = len(hs_bl)
    tokenizer = processor.tokenizer

    # BL self-metrics
    sve_bl = [
        round(compute_sve(hs_bl[li]), 4)
        for li in range(n_layers)
    ]
    norm_bl = [
        round(compute_l2_norm(hs_bl[li]), 4)
        for li in range(n_layers)
    ]
    entropy_bl = round(compute_entropy(logits_bl), 4)
    top1_bl = get_top1_token(logits_bl, tokenizer)

    # Per-condition metrics
    conditions = {
        "sm": (hs_sm, logits_sm, True),
        "he": (hs_he, logits_he, n_sink == 0),
        "pm": (hs_pm, logits_pm, True),
        "anti": (hs_anti, logits_anti, True),
        "r50": (hs_r50, logits_r50, True),
        "r90": (hs_r90, logits_r90, True),
        "to": (hs_to, logits_to, False),
    }

    metrics = {}
    for cond, (hs_c, log_c, same_len) in conditions.items():
        m = _metrics_for_pair(
            hs_bl, hs_c, logits_bl, log_c,
            tokenizer, n_layers, same_len,
        )
        metrics[cond] = m

    # Build result dict
    result = {
        "n_vis": n_vis,
        "n_sink": n_sink,
        "n_anti": n_vis - n_sink,
        "sink_frac": round(n_sink / n_vis if n_vis else 0, 3),
        "seq_len": int(seq_len),
        "n_layers": n_layers,
        "n_mask_50": n_mask_50,
        "n_mask_90": n_mask_90,
        "sve_bl": sve_bl,
        "norm_bl": norm_bl,
        "entropy_bl": entropy_bl,
        "top1_bl": top1_bl,
    }
    for cond, m in metrics.items():
        result[f"sve_{cond}"] = m["sve"]
        result[f"norm_{cond}"] = m["norm"]
        result[f"cosine_bl_{cond}"] = m["cosine"]
        result[f"kl_bl_{cond}"] = m["kl"]
        result[f"entropy_{cond}"] = m["entropy"]
        result[f"topk_retention_{cond}"] = m["topk"]
        result[f"top1_{cond}"] = m["top1"]

    raw = {
        "hs_bl": hs_bl, "logits_bl": logits_bl,
        "hs_sm": hs_sm, "logits_sm": logits_sm,
        "hs_r50": hs_r50, "logits_r50": logits_r50,
        "hs_r90": hs_r90, "logits_r90": logits_r90,
    }

    return result, raw


# ── Main ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Representation analysis: sink masking "
                    "effects on hidden states & logits",
    )
    ap.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-VL-7B-Instruct-8bit",
    )
    ap.add_argument(
        "--benchmarks", nargs="+",
        default=["mmstar", "pope"],
        choices=["mmstar", "pope", "scienceqa"],
    )
    ap.add_argument(
        "--max-per-benchmark", type=int, default=50,
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tau", type=float, default=20.0)
    ap.add_argument("--detect-layer", type=int, default=14)
    ap.add_argument(
        "--save-raw", type=int, default=0,
        help="Save raw hidden states for N samples (for "
             "detailed figures). 0 = no raw saves.",
    )
    ap.add_argument(
        "--output-dir",
        default="../../experiments/repr_analysis/output",
    )
    ap.add_argument(
        "--discover-dims", action="store_true",
        help="Auto-discover sink dims on first sample",
    )
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_raw > 0:
        os.makedirs(
            os.path.join(args.output_dir, "raw"), exist_ok=True,
        )

    sink_dims = [458, 2570]

    # ── Init logger ────────────────────────────────────
    log_path = os.path.join(args.output_dir, "exp_run.jsonl")
    lg = ExpLogger(log_path)
    lg.log(
        "exp_start",
        experiment="representation_analysis",
        model=args.model,
        seed=args.seed,
        benchmarks=args.benchmarks,
        max_per_benchmark=args.max_per_benchmark,
        tau=args.tau,
        detect_layer=args.detect_layer,
        save_raw=args.save_raw,
        sink_dims=sink_dims,
    )

    # ── Load model ─────────────────────────────────────
    from mlx_vlm import load
    from exp_three_way_masking import (
        patch_model_v2,
        prepare_vision_input,
    )
    from sink_detect import (
        find_image_token_range,
        discover_sink_dims,
        CaptureStore,
    )
    from benchmark_loaders import (
        load_mmstar,
        load_pope,
        load_scienceqa_img,
    )

    print(f"Loading {args.model}...")
    lg.log("model_load_start", model=args.model)
    model, processor = load(args.model)
    patch_model_v2(model)
    mcfg = model.config
    rng = np.random.RandomState(args.seed)
    lg.log("model_load_done")

    # ── Benchmark loader dispatch ──────────────────────
    def load_benchmark(name, max_samples, seed):
        if name == "mmstar":
            return load_mmstar(max_samples, seed)
        elif name == "pope":
            return load_pope(max_samples, seed)
        elif name == "scienceqa":
            return load_scienceqa_img(max_samples, seed)
        else:
            raise ValueError(f"Unknown benchmark: {name}")

    # ── Run benchmarks ─────────────────────────────────
    t0_global = time.time()
    dims_discovered = False
    raw_saved = 0

    for bname in args.benchmarks:
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {bname}")
        print(f"{'='*60}")

        lg.log("bench_start", benchmark=bname)
        items = load_benchmark(
            bname, args.max_per_benchmark, args.seed,
        )
        print(f"  Loaded {len(items)} items")
        lg.log("bench_loaded", benchmark=bname,
               n_items=len(items))

        if not items:
            print("  SKIP: no items")
            continue

        for qi, item in enumerate(items):
            try:
                q_start = time.time()

                # Resolve lazy image
                image = item["image"]
                if image is None and "_load_image" in item:
                    image = item["_load_image"]()

                ids, pv, attn_mask, extra = (
                    prepare_vision_input(
                        processor, item["question"],
                        item["candidates"],
                        image=image,
                    )
                )

                # Auto-discover sink dims on first sample
                if args.discover_dims and not dims_discovered:
                    print("  Discovering sink dims...")
                    hs_disc, _ = forward_with_capture(
                        model, ids, pv, attn_mask, extra,
                    )
                    # discover_sink_dims expects 3D [B,L,D]
                    hs_disc_3d = [
                        h[np.newaxis, :, :] for h in hs_disc
                    ]
                    img_dims, _ = discover_sink_dims(hs_disc_3d)
                    print(f"  Discovered: {img_dims}")
                    if img_dims != sink_dims:
                        print(
                            f"  Using discovered dims "
                            f"(was {sink_dims})"
                        )
                        sink_dims = img_dims
                    dims_discovered = True
                    del hs_disc
                    gc.collect()

                # Run analysis
                result, raw = analyze_sample(
                    model, processor, ids, pv,
                    attn_mask, extra, mcfg,
                    sink_dims, args.tau,
                    args.detect_layer, rng,
                    question=item["question"],
                )

                q_elapsed = time.time() - q_start

                # Log to JSONL
                lg.log(
                    "sample",
                    benchmark=bname,
                    qi=qi,
                    elapsed_sec=round(q_elapsed, 2),
                    **result,
                )

                # Save raw data for detailed figures
                if (
                    args.save_raw > 0
                    and raw_saved < args.save_raw
                ):
                    raw_path = os.path.join(
                        args.output_dir, "raw",
                        f"raw_{bname}_{qi:04d}.npz",
                    )
                    # Save SVD spectra for representative layers
                    svd_spectra = {}
                    for li in [0, 7, 14, 21, 27]:
                        if li < len(raw["hs_bl"]):
                            for cond, key in [
                                ("bl", "hs_bl"),
                                ("sm", "hs_sm"),
                                ("r50", "hs_r50"),
                                ("r90", "hs_r90"),
                            ]:
                                try:
                                    sv = np.linalg.svd(
                                        raw[key][li].astype(
                                            np.float64,
                                        ),
                                        compute_uv=False,
                                    )
                                    svd_spectra[
                                        f"svd_{cond}_L{li}"
                                    ] = sv.astype(np.float32)
                                except np.linalg.LinAlgError:
                                    pass

                    # Per-token cosine for heatmap
                    cosine_arrays = {}
                    for cond, key in [
                        ("sm", "hs_sm"),
                        ("r50", "hs_r50"),
                        ("r90", "hs_r90"),
                    ]:
                        cos_list = []
                        for li in range(len(raw["hs_bl"])):
                            _, cos = compute_cosine_sim(
                                raw["hs_bl"][li],
                                raw[key][li],
                            )
                            cos_list.append(cos)
                        cosine_arrays[
                            f"cosine_per_token_{cond}"
                        ] = np.array(cos_list)

                    np.savez_compressed(
                        raw_path,
                        logits_bl=raw["logits_bl"],
                        logits_sm=raw["logits_sm"],
                        logits_r50=raw["logits_r50"],
                        logits_r90=raw["logits_r90"],
                        **cosine_arrays,
                        **svd_spectra,
                    )
                    print(f"    Saved raw -> {raw_path}")
                    raw_saved += 1

                # Free memory
                del raw
                gc.collect()
                mx.metal.clear_cache()

                # Print status
                kls = "  ".join([
                    f"{c}={result.get(f'kl_bl_{c}', 0):.1f}"
                    for c in ["sm", "he", "pm", "anti",
                              "r50", "r90", "to"]
                ])
                print(
                    f"  [{qi+1}/{len(items)}] "
                    f"sink={result['n_sink']}/{result['n_vis']}  "
                    f"KL: {kls}  "
                    f"{q_elapsed:.1f}s"
                )

            except Exception as ex:
                print(f"  ERROR Q{qi}: {ex}")
                lg.log("error", benchmark=bname, qi=qi,
                       msg=str(ex))
                import traceback
                traceback.print_exc()

        # Benchmark summary
        lg.log("bench_end", benchmark=bname)

    # ── Done ───────────────────────────────────────────
    total_elapsed = time.time() - t0_global
    print(f"\nTotal time: {total_elapsed/60:.1f} min")
    lg.log(
        "exp_end",
        total_min=round(total_elapsed / 60, 1),
    )
    lg.close()
    print(f"Results -> {log_path}")


if __name__ == "__main__":
    main()
