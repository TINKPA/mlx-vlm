"""
Three-way masking ablation: baseline vs soft-mask vs hard-evict.

Answers: does sink's FFN/residual path contribute useful info?
  - soft mask ≈ hard evict → FFN also useless, hard evict is optimal
  - soft mask > hard evict → FFN path has value, keep it

Protocol per question:
  1. Baseline: normal inference + capture hidden states → detect sinks
  2. Soft mask: re-run with attention blocking sink columns
  3. Hard evict: re-run with sink tokens removed from embeddings

Exports for multi-benchmark use:
  - prepare_vision_input(processor, question, candidates,
                         image=None, video_path=None)
  - run_all_ablations(model, processor, ids, pv, attn_mask,
                      extra, mcfg, sink_dims, tau,
                      detect_layer, rng, max_tokens)

Usage:
  uv run --with "transformers<4.52" --with torch \
    --with matplotlib \
    python exp_three_way_masking.py \
      --data-dir /Volumes/RAID0/datasets/MVBench \
      --tasks action_antonym \
      --max-questions 20 \
      --output-dir exp_three_way_output
"""

import argparse
import json
import os
import time
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sink_detect import (
    CaptureStore,
    find_image_token_range,
    rmsnorm,
)
from sink_eviction_v9_streaming import generate_from_cache
from sink_eviction_v10_mvbench import (
    load_mvbench_tasks,
    find_video,
    format_mcq_prompt,
    extract_answer_mvbench,
)


# ── Soft Mask infrastructure ──────────────────────────

class SoftMask:
    """Column mask to block attention TO specific positions."""
    mask_1d = None   # shape [max_seq_len], 0 or -inf
    enabled = False

    @classmethod
    def set_blocked(cls, positions, seq_len):
        m = np.zeros(seq_len, dtype=np.float32)
        for p in positions:
            if p < seq_len:
                m[p] = -1e9  # large negative, not -inf to avoid NaN
        cls.mask_1d = mx.array(m)
        cls.enabled = True

    @classmethod
    def disable(cls):
        cls.enabled = False
        cls.mask_1d = None


# ── Patching ──────────────────────────────────────────

_orig_attn_call = None
_orig_decoder_call = None


def _patched_decoder_call(self, x, mask=None, cache=None,
                          position_ids=None):
    """Capture hidden states before attention."""
    if CaptureStore.enabled and x.shape[1] > 1:
        CaptureStore.hidden_states.append(x)
    return _orig_decoder_call(
        self, x, mask=mask, cache=cache,
        position_ids=position_ids,
    )


def _patched_attn_call(self, x, mask=None, cache=None,
                       position_ids=None):
    """Attention with optional soft mask + capture."""
    B, L, D = x.shape

    queries = self.q_proj(x)
    keys = self.k_proj(x)
    values = self.v_proj(x)

    queries = queries.reshape(
        B, L, self.n_heads, self.head_dim,
    ).transpose(0, 2, 1, 3)
    keys = keys.reshape(
        B, L, self.n_kv_heads, self.head_dim,
    ).transpose(0, 2, 1, 3)
    values = values.reshape(
        B, L, self.n_kv_heads, self.head_dim,
    ).transpose(0, 2, 1, 3)

    if position_ids is None:
        kv_seq_len = keys.shape[-2]
        kv_seq_len += cache.offset + 1
        position_ids = mx.arange(
            cache.offset, cache.offset + L,
        )
        position_ids = mx.expand_dims(position_ids, axis=0)
        position_ids = mx.tile(position_ids, (3, 1, 1))
    else:
        kv_seq_len = keys.shape[-2]
        kv_seq_len += (
            cache.offset + 1 if cache is not None else 0
        )

    from mlx_vlm.models.qwen2_5_vl.language import (
        apply_multimodal_rotary_pos_emb,
    )
    cos, sin = self.rotary_emb(values, position_ids)

    if mask is not None and isinstance(mask, mx.array):
        mask = mask[..., :keys.shape[-2]]

    queries, keys = apply_multimodal_rotary_pos_emb(
        queries, keys, cos, sin, unqueeze_dim=1,
    )

    if cache is not None:
        keys, values = cache.update_and_fetch(keys, values)

    n_rep = self.n_heads // self.n_kv_heads
    if n_rep > 1:
        keys = mx.repeat(keys, n_rep, axis=1)
        values = mx.repeat(values, n_rep, axis=1)

    # Decide if manual attention is needed
    need_manual = (
        (CaptureStore.enabled and L > 1)
        or SoftMask.enabled
    )

    if need_manual:
        attn_scores = (
            queries @ keys.transpose(0, 1, 3, 2)
        ) * self.scale

        if mask is not None and isinstance(mask, mx.array):
            attn_scores = attn_scores + mask
        elif mask is not None and mask == "causal":
            L_q = attn_scores.shape[2]
            L_k = attn_scores.shape[3]
            causal = mx.triu(
                mx.full((L_q, L_k), float("-inf")), k=1,
            )
            attn_scores = attn_scores + causal

        # Apply soft mask (block sink columns)
        if SoftMask.enabled and SoftMask.mask_1d is not None:
            K = attn_scores.shape[-1]
            sm_len = SoftMask.mask_1d.shape[0]
            if K <= sm_len:
                sm = SoftMask.mask_1d[:K]
            else:
                # During generation, K grows beyond prefill
                # length. Pad with 0 (don't block new tokens).
                pad = mx.zeros((K - sm_len,))
                sm = mx.concatenate(
                    [SoftMask.mask_1d, pad],
                )
            attn_scores = attn_scores + sm.reshape(1, 1, 1, -1)

        attn_w = mx.softmax(attn_scores, axis=-1)
        if CaptureStore.enabled and L > 1:
            CaptureStore.attn_weights.append(attn_w)
        output = attn_w @ values
    else:
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values,
            scale=self.scale, mask=mask,
        )

    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(output)


def patch_model_v2(model):
    """Patch with capture + soft mask support."""
    global _orig_attn_call, _orig_decoder_call

    from mlx_vlm.models.qwen2_5_vl.language import (
        Attention, Qwen2VLDecoderLayer,
    )

    _orig_decoder_call = Qwen2VLDecoderLayer.__call__
    Qwen2VLDecoderLayer.__call__ = _patched_decoder_call

    _orig_attn_call = Attention.__call__
    Attention.__call__ = _patched_attn_call

    n = len(model.language_model.model.layers)
    print(f"Patched {n} layers (capture + soft mask).")


# ── Sink detection ────────────────────────────────────

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
        np.stack([rms_val[:, d] for d in sink_dims], axis=-1),
        axis=-1,
    )
    vis_phi = phi[img_start:img_end]
    sink_local = np.where(vis_phi > tau)[0]
    sink_abs = sink_local + img_start
    return sink_abs, vis_phi


# ── Inference helpers ─────────────────────────────────

def run_baseline(model, processor, ids, pv, mask, extra,
                 max_tok=30):
    """Normal inference with hidden state capture."""
    from mlx_vlm.models.cache import make_prompt_cache

    CaptureStore.enable()
    SoftMask.disable()

    cache = make_prompt_cache(model.language_model)
    eo = model.get_input_embeddings(
        ids, pv, mask=mask, **extra,
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

    hs_np = [np.array(h) for h in CaptureStore.hidden_states]
    CaptureStore.disable()

    # Also save position_ids and embeds for reuse
    pos_ids = model.language_model._position_ids
    rope_deltas = model.language_model._rope_deltas

    text = generate_from_cache(
        model, processor, cache, logits, max_tok,
    )
    return text, hs_np, embeds, pos_ids, rope_deltas, fkw


def run_soft_mask(model, processor, ids, pv, mask, extra,
                  sink_abs, seq_len, max_tok=30):
    """Inference with attention column masking on sinks."""
    from mlx_vlm.models.cache import make_prompt_cache

    CaptureStore.disable()
    SoftMask.set_blocked(sink_abs.tolist(), seq_len)

    # Reset model state so position_ids are recomputed
    model.language_model._position_ids = None
    model.language_model._rope_deltas = None

    cache = make_prompt_cache(model.language_model)
    eo = model.get_input_embeddings(
        ids, pv, mask=mask, **extra,
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

    text = generate_from_cache(
        model, processor, cache, logits, max_tok,
    )
    SoftMask.disable()
    return text


def run_hard_evict(model, processor, ids, embeds, pos_ids,
                   rope_deltas, sink_abs, max_tok=30):
    """Inference with sink tokens removed from sequence."""
    from mlx_vlm.models.cache import make_prompt_cache

    CaptureStore.disable()
    SoftMask.disable()

    seq_len = embeds.shape[1]

    # Create integer keep indices (MLX doesn't support bool indexing)
    keep = np.ones(seq_len, dtype=bool)
    for p in sink_abs:
        if p < seq_len:
            keep[p] = False
    keep_idx = mx.array(np.where(keep)[0])

    # Remove sink tokens from embeddings
    evicted_embeds = embeds[:, keep_idx, :]
    evicted_len = evicted_embeds.shape[1]

    # Remove from position_ids: shape (3, batch, seq_len)
    if pos_ids is not None:
        evicted_pos = pos_ids[:, :, keep_idx]
    else:
        evicted_pos = None

    # Create dummy ids of correct length
    evicted_ids = ids[:, keep_idx]

    # Set model state for generation
    model.language_model._position_ids = evicted_pos
    if evicted_pos is not None:
        # rope_deltas = max_position + 1 - seq_length
        max_pos = evicted_pos.max()
        model.language_model._rope_deltas = (
            max_pos + 1 - evicted_len
        )
    else:
        model.language_model._rope_deltas = rope_deltas

    cache = make_prompt_cache(model.language_model)

    # Call inner model directly with explicit position_ids
    h = model.language_model.model(
        evicted_ids,
        cache=cache,
        inputs_embeds=evicted_embeds,
        position_ids=evicted_pos,
    )
    mx.eval([c.state for c in cache])

    # Compute logits
    if model.language_model.args.tie_word_embeddings:
        logits = model.language_model.model.embed_tokens.as_linear(h)
    else:
        logits = model.language_model.lm_head(h)
    logits = logits[:, -1, :]

    text = generate_from_cache(
        model, processor, cache, logits, max_tok,
    )
    return text


# ── Reusable helpers for multi-benchmark ──────────────

def prepare_vision_input(
    processor, question, candidates=None,
    image=None, video_path=None,
):
    """
    Build model inputs for a single question.

    Returns (ids, pv, attn_mask, extra) as mx.arrays.
    Works for both image and video inputs.
    If neither image nor video_path is given, returns
    text-only inputs (pv=None).
    """
    from mlx_vlm.video_generate import process_vision_info

    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    elif video_path is not None:
        content.append({
            "type": "video",
            "video": video_path,
            "max_pixels": 224 * 224,
            "fps": 1.0,
        })
    content.append({"type": "text", "text": question})

    msgs = [{"role": "user", "content": content}]
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
    # Determine pixel values key
    pv = None
    for pvk in [
        "pixel_values_videos", "pixel_values",
    ]:
        if inputs.get(pvk) is not None:
            pv = mx.array(inputs[pvk])
            break

    attn_mask = mx.array(inputs["attention_mask"])
    extra = {}
    for k in ["video_grid_thw", "image_grid_thw"]:
        if inputs.get(k) is not None:
            extra[k] = mx.array(inputs[k])

    return ids, pv, attn_mask, extra


def run_all_ablations(
    model, processor, ids, pv, attn_mask, extra,
    mcfg, sink_dims, tau=20.0, detect_layer=14,
    rng=None, max_tokens=30,
):
    """
    Run all 5 ablation conditions on a single question.

    Returns dict:
      out_bl, out_sm, out_he, out_rand, out_anti: str
      pred_bl, pred_sm, pred_he, pred_rand, pred_anti: str
      n_vis, n_sink, sink_frac: int/float
      sink_abs, vis_phi: np.ndarray
    """
    if rng is None:
        rng = np.random.RandomState(42)

    s, e = find_image_token_range(ids, mcfg)
    seq_len = ids.shape[1]
    n_vis = e - s

    # ── Baseline ──────────────────────────────────────
    (out_bl, hs_np, embeds, pos_ids,
     rope_deltas, fkw) = run_baseline(
        model, processor, ids, pv,
        attn_mask, extra, max_tokens,
    )

    # Detect sinks
    sink_abs, vis_phi = detect_sinks(
        hs_np, s, e, sink_dims, detect_layer, tau,
    )
    n_sink = len(sink_abs)

    # ── Soft mask sink ────────────────────────────────
    if n_sink > 0:
        out_sm = run_soft_mask(
            model, processor, ids, pv,
            attn_mask, extra,
            sink_abs, seq_len, max_tokens,
        )
    else:
        out_sm = out_bl

    # ── Hard evict ────────────────────────────────────
    if n_sink > 0:
        model.language_model._position_ids = None
        model.language_model._rope_deltas = None
        eo2 = model.get_input_embeddings(
            ids, pv, mask=attn_mask, **extra,
        )
        embeds2 = eo2.inputs_embeds
        pos_ids2 = model.language_model._position_ids
        rope_deltas2 = model.language_model._rope_deltas
        out_he = run_hard_evict(
            model, processor, ids,
            embeds2, pos_ids2, rope_deltas2,
            sink_abs, max_tokens,
        )
    else:
        out_he = out_bl

    # ── SM-random (control) ───────────────────────────
    if n_sink > 0:
        vis_indices = np.arange(s, e)
        rand_sel = rng.choice(
            vis_indices, size=n_sink, replace=False,
        )
        out_rand = run_soft_mask(
            model, processor, ids, pv,
            attn_mask, extra,
            rand_sel, seq_len, max_tokens,
        )
    else:
        out_rand = out_bl

    # ── SM-anti-sink (control) ────────────────────────
    if n_sink > 0 and n_sink < n_vis:
        all_vis = set(range(s, e))
        sink_set = set(sink_abs.tolist())
        anti_sink = np.array(sorted(all_vis - sink_set))
        out_anti = run_soft_mask(
            model, processor, ids, pv,
            attn_mask, extra,
            anti_sink, seq_len, max_tokens,
        )
    else:
        out_anti = out_bl

    # Clean up for memory
    del hs_np, embeds, fkw
    mx.metal.clear_cache()

    return {
        "out_bl": out_bl,
        "out_sm": out_sm,
        "out_he": out_he,
        "out_rand": out_rand,
        "out_anti": out_anti,
        "n_vis": n_vis,
        "n_sink": n_sink,
        "sink_frac": round(
            n_sink / n_vis if n_vis else 0, 3,
        ),
        "sink_abs": sink_abs,
        "vis_phi": vis_phi,
        "img_start": s,
        "img_end": e,
    }


# ── Main ──────────────────────────────────────────────

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
        "--output-dir", default="exp_three_way_output",
    )
    ap.add_argument(
        "--max-questions", type=int, default=20,
    )
    ap.add_argument(
        "--tasks", type=str, nargs="+", default=None,
    )
    ap.add_argument("--tau", type=float, default=20.0)
    ap.add_argument("--detect-layer", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sink_dims = [458, 2570]

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
    from mlx_vlm.video_generate import process_vision_info

    print(f"\nLoading {args.model}...")
    model, processor = load(args.model)
    patch_model_v2(model)
    mcfg = model.config

    # ── Evaluate ──────────────────────────────────────
    results = []
    t0 = time.time()

    for tname in sorted(sampled.keys()):
        questions = sampled[tname]
        print(f"\n--- {tname} ({len(questions)} Qs) ---")

        for qi, item in enumerate(questions):
            video_name = item["video"]
            question = item["question"]
            candidates = item["candidates"]
            gt = item["answer"]

            video_path = find_video(
                video_name, args.data_dir,
            )
            if video_path is None:
                print(f"  SKIP Q{qi}: video not found")
                continue

            try:
                q_start = time.time()

                # Prepare input
                prompt = format_mcq_prompt(
                    question, candidates,
                )
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
                img_in, vid_in, _ = process_vision_info(
                    msgs, True,
                )
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
                attn_mask = mx.array(inputs["attention_mask"])
                extra = {}
                for k in [
                    "video_grid_thw", "image_grid_thw",
                ]:
                    if inputs.get(k) is not None:
                        extra[k] = mx.array(inputs[k])

                s, e = find_image_token_range(ids, mcfg)
                seq_len = ids.shape[1]

                # ── Baseline ──────────────────────────
                (out_bl, hs_np, embeds, pos_ids,
                 rope_deltas, fkw) = run_baseline(
                    model, processor, ids, pv,
                    attn_mask, extra, args.max_tokens,
                )
                pred_bl, _ = extract_answer_mvbench(
                    out_bl, candidates,
                )

                # Detect sinks
                sink_abs, vis_phi = detect_sinks(
                    hs_np, s, e, sink_dims,
                    args.detect_layer, args.tau,
                )
                n_sink = len(sink_abs)
                n_vis = e - s

                # ── Soft mask ─────────────────────────
                if n_sink > 0:
                    out_sm = run_soft_mask(
                        model, processor, ids, pv,
                        attn_mask, extra,
                        sink_abs, seq_len,
                        args.max_tokens,
                    )
                else:
                    out_sm = out_bl  # no sinks, same result
                pred_sm, _ = extract_answer_mvbench(
                    out_sm, candidates,
                )

                # ── Hard evict ────────────────────────
                if n_sink > 0:
                    # Reset model state
                    model.language_model._position_ids = None
                    model.language_model._rope_deltas = None
                    # Re-get embeddings + position_ids
                    eo2 = model.get_input_embeddings(
                        ids, pv, mask=attn_mask, **extra,
                    )
                    embeds2 = eo2.inputs_embeds
                    pos_ids2 = (
                        model.language_model._position_ids
                    )
                    rope_deltas2 = (
                        model.language_model._rope_deltas
                    )

                    out_he = run_hard_evict(
                        model, processor, ids,
                        embeds2, pos_ids2, rope_deltas2,
                        sink_abs, args.max_tokens,
                    )
                else:
                    out_he = out_bl
                pred_he, _ = extract_answer_mvbench(
                    out_he, candidates,
                )

                # ── SM-random (control) ───────────────
                if n_sink > 0:
                    # Random tokens from visual range,
                    # same count as sinks
                    vis_indices = np.arange(s, e)
                    rand_sel = rng.choice(
                        vis_indices, size=n_sink,
                        replace=False,
                    )
                    out_rand = run_soft_mask(
                        model, processor, ids, pv,
                        attn_mask, extra,
                        rand_sel, seq_len,
                        args.max_tokens,
                    )
                else:
                    out_rand = out_bl
                pred_rand, _ = extract_answer_mvbench(
                    out_rand, candidates,
                )

                # ── SM-anti-sink (control) ────────────
                if n_sink > 0 and n_sink < n_vis:
                    # Mask the NON-sink tokens (keep only
                    # sinks), destroying content
                    all_vis = set(range(s, e))
                    sink_set = set(sink_abs.tolist())
                    anti_sink = np.array(
                        sorted(all_vis - sink_set),
                    )
                    out_anti = run_soft_mask(
                        model, processor, ids, pv,
                        attn_mask, extra,
                        anti_sink, seq_len,
                        args.max_tokens,
                    )
                else:
                    out_anti = out_bl
                pred_anti, _ = extract_answer_mvbench(
                    out_anti, candidates,
                )

                q_elapsed = time.time() - q_start
                correct_bl = (pred_bl == gt)
                correct_sm = (pred_sm == gt)
                correct_he = (pred_he == gt)
                correct_rand = (pred_rand == gt)
                correct_anti = (pred_anti == gt)

                result = {
                    "task": tname,
                    "qi": qi,
                    "gt": gt,
                    "elapsed_sec": round(q_elapsed, 2),
                    "n_vis": n_vis,
                    "n_sink": n_sink,
                    "sink_frac": round(
                        n_sink / n_vis if n_vis else 0, 3,
                    ),
                    # Baseline
                    "pred_bl": pred_bl,
                    "correct_bl": correct_bl,
                    "out_bl": out_bl[:80],
                    # Soft mask sink
                    "pred_sm": pred_sm,
                    "correct_sm": correct_sm,
                    "out_sm": out_sm[:80],
                    # Hard evict
                    "pred_he": pred_he,
                    "correct_he": correct_he,
                    "out_he": out_he[:80],
                    # SM-random
                    "pred_rand": pred_rand,
                    "correct_rand": correct_rand,
                    "out_rand": out_rand[:80],
                    # SM-anti-sink
                    "pred_anti": pred_anti,
                    "correct_anti": correct_anti,
                    "out_anti": out_anti[:80],
                    # Comparison
                    "bl_eq_sm": (pred_bl == pred_sm),
                    "bl_eq_he": (pred_bl == pred_he),
                    "sm_eq_he": (pred_sm == pred_he),
                    "bl_eq_rand": (pred_bl == pred_rand),
                    "bl_eq_anti": (pred_bl == pred_anti),
                }
                results.append(result)

                m_bl = "O" if correct_bl else "X"
                m_sm = "O" if correct_sm else "X"
                m_he = "O" if correct_he else "X"
                m_rn = "O" if correct_rand else "X"
                m_an = "O" if correct_anti else "X"
                print(
                    f"  Q{qi}: BL[{m_bl}] SM[{m_sm}] "
                    f"HE[{m_he}] RN[{m_rn}] AN[{m_an}]  "
                    f"sink={n_sink}/{n_vis}"
                )

            except Exception as ex:
                print(f"  ERROR Q{qi}: {ex}")
                import traceback
                traceback.print_exc()

    # ── Summary ───────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"THREE-WAY MASKING ABLATION")
    print(f"  {len(results)} questions in "
          f"{elapsed/60:.1f} min")
    print(f"{'='*60}")

    if results:
        n = len(results)
        acc_bl = sum(r["correct_bl"] for r in results) / n
        acc_sm = sum(r["correct_sm"] for r in results) / n
        acc_he = sum(r["correct_he"] for r in results) / n
        acc_rn = sum(r["correct_rand"] for r in results) / n
        acc_an = sum(r["correct_anti"] for r in results) / n
        sm_eq_he = sum(r["sm_eq_he"] for r in results)
        avg_sink = np.mean(
            [r["sink_frac"] for r in results]
        )

        print(f"\nAccuracy:")
        print(f"  Baseline:      {acc_bl:.1%}")
        print(f"  SM-sink:       {acc_sm:.1%}")
        print(f"  Hard evict:    {acc_he:.1%}")
        print(f"  SM-random:     {acc_rn:.1%}")
        print(f"  SM-anti-sink:  {acc_an:.1%}")

        print(f"\nAgreement:")
        print(f"  SM == HE: {sm_eq_he}/{n} ({sm_eq_he/n:.1%})")

        print(f"\nAvg sink fraction: {avg_sink:.1%}")

        print(f"\nInterpretation:")
        # φ signal value
        if acc_sm > acc_rn + 0.02:
            print(f"  SM-sink ({acc_sm:.1%}) > SM-random "
                  f"({acc_rn:.1%}) → φ identifies SAFER "
                  f"tokens to remove")
        elif acc_sm < acc_rn - 0.02:
            print(f"  SM-sink ({acc_sm:.1%}) < SM-random "
                  f"({acc_rn:.1%}) → φ signal NOT useful")
        else:
            print(f"  SM-sink ≈ SM-random → removing ANY "
                  f"tokens is equally (un)harmful")
        # Anti-sink
        if acc_an < 0.30:
            print(f"  SM-anti-sink ({acc_an:.1%}) → near "
                  f"chance level. Content tokens carry "
                  f"ALL the information.")
        else:
            print(f"  SM-anti-sink ({acc_an:.1%}) → "
                  f"surprisingly high. Model robust to "
                  f"content removal.")

    # Save
    jp = os.path.join(args.output_dir, "results.json")
    with open(jp, "w") as f:
        json.dump({
            "experiment": "three_way_masking",
            "seed": args.seed,
            "detect_layer": args.detect_layer,
            "tau": args.tau,
            "sink_dims": sink_dims,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved → {jp}")


if __name__ == "__main__":
    main()
