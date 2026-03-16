"""
v9: Streaming KV cache eviction during prefill.

Key insight from v7: post-prefill eviction doesn't affect factual
accuracy because visual info is encoded into text hidden states
during prefill. This experiment tests the REAL scenario: eviction
DURING prefill, so text tokens never see evicted visual tokens.

Protocol:
  1. Process system tokens (before visual) → cache
  2. Process visual tokens chunk-by-chunk (simulating frames)
  3. After each chunk, if visual cache > budget, score & evict
  4. Process text tokens on the reduced visual cache
  5. Generate answer → MCQ accuracy

This is where sink-aware eviction should matter: choosing WHICH
visual tokens to keep when the model hasn't seen them all yet.

Usage:
  uv run --with "transformers<4.52" --with torch \
    --with matplotlib \
    python sink_eviction_v9_streaming.py \
      --output-dir sink_eviction_output_v9
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from sink_detect import (
    CaptureStore,
    find_image_token_range,
    patch_model,
    rmsnorm,
    score_phi_multilayer,
)


# ── MCQ Benchmark (same as v7) ──────────────────────────
BENCHMARK = [
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


# ── Scoring functions ───────────────────────────────────

def score_phi_chunk(
    hidden_states: List,
    chunk_start_in_seq: int,
    chunk_len: int,
    dims: List[int] = None,
    layer: int = 14,
) -> np.ndarray:
    """
    Compute φ scores for tokens in the current chunk.
    hidden_states: captured during this chunk's forward pass.
    Returns array of shape [chunk_len].
    """
    if dims is None:
        dims = [458, 2570]
    li = min(layer, len(hidden_states) - 1)
    h = np.array(hidden_states[li])
    if h.ndim == 3:
        h = h[0]  # [seq_len, hidden_dim]
    rms = np.abs(rmsnorm(h))
    # h contains only the chunk's tokens
    phi = np.max(
        np.stack([rms[:, d] for d in dims], -1), -1,
    )
    return phi


def score_phi_multilayer_chunk(
    hidden_states: List,
    chunk_start_in_seq: int,
    chunk_len: int,
    dims: List[int] = None,
    layers: List[int] = None,
    weights: List[float] = None,
) -> np.ndarray:
    """
    Compute multi-layer aggregated φ scores for tokens in the
    current chunk.  Analogous to score_phi_chunk() but uses
    score_phi_multilayer() across multiple layers.

    hidden_states: captured during this chunk's forward pass.
    Returns array of shape [chunk_len].
    """
    if dims is None:
        dims = [458, 2570]
    if layers is None:
        layers = [14, 18, 22, 24]
    return score_phi_multilayer(
        hidden_states,
        token_indices=None,  # chunk hs already scoped
        layers=layers,
        weights=weights,
        sink_dims=dims,
    )


def score_attn_chunk(
    attn_weights: List,
    chunk_len: int,
    grounding_layers: List[int] = None,
) -> np.ndarray:
    """
    Compute attention-based importance for chunk tokens.

    Each chunk's attn_weights[li] has shape
    [1, n_heads, chunk_len, cache_len + chunk_len].
    We measure how much attention each chunk token receives
    (summed over all query positions in the chunk), averaged
    across grounding layers.

    Returns array of shape [chunk_len] — higher means the
    token receives more attention (more important to keep).
    """
    if grounding_layers is None:
        grounding_layers = [18, 22, 24]

    scores = np.zeros(chunk_len)
    n_used = 0

    for li in grounding_layers:
        if li >= len(attn_weights):
            continue
        aw = np.array(attn_weights[li])
        if aw.ndim == 4:
            aw = aw[0]  # [n_heads, Q, K]
        # Average over heads
        aw_mean = aw.mean(axis=0)  # [Q, K]
        # Q = chunk_len, K = (previous_cache + chunk_len)
        # The last chunk_len columns of K correspond to
        # this chunk's own tokens.
        k_len = aw_mean.shape[1]
        chunk_cols = aw_mean[:, k_len - chunk_len:]
        # Sum incoming attention to each chunk token
        # across all query positions in this chunk
        scores += chunk_cols.sum(axis=0)
        n_used += 1

    if n_used > 0:
        scores /= n_used
    return scores


def _normalize_minmax(x: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    mn, mx_ = x.min(), x.max()
    if mx_ - mn < 1e-10:
        return np.ones_like(x) * 0.5
    return (x - mn) / (mx_ - mn)


def score_hybrid_chunk(
    capture_store,
    chunk_len: int,
    dims: List[int] = None,
    phi_layer: int = 14,
    alpha: float = 0.5,
    grounding_layers: List[int] = None,
) -> np.ndarray:
    """
    Hybrid score combining attention and phi (sink dim).

    HIGH attention + LOW phi = content token = HIGH score.
    score = alpha * attn_norm + (1-alpha) * (1 - phi_norm)

    The phi-attn correlation is only ~0.195 (nearly
    orthogonal), so combining them captures complementary
    information about token importance.

    Args:
        capture_store: CaptureStore with .hidden_states
            and .attn_weights from the current chunk.
        chunk_len: number of tokens in this chunk.
        dims: sink dimensions for phi scoring.
        phi_layer: which layer for phi scoring.
        alpha: weight for attention component (0-1).
        grounding_layers: layers to average attention
            scores from.

    Returns:
        Per-token hybrid scores, shape [chunk_len].
        Higher score = more worth keeping.
    """
    if grounding_layers is None:
        grounding_layers = [18, 22, 24]

    # phi scores from hidden states
    phi = score_phi_chunk(
        capture_store.hidden_states, 0, chunk_len,
        dims=dims, layer=phi_layer,
    )
    # Attention scores from attn_weights
    attn = score_attn_chunk(
        capture_store.attn_weights, chunk_len,
        grounding_layers=grounding_layers,
    )

    # Normalize both to [0, 1]
    phi_norm = _normalize_minmax(phi)
    attn_norm = _normalize_minmax(attn)

    # Combine: high attn + low phi = content = KEEP
    score = (
        alpha * attn_norm
        + (1 - alpha) * (1 - phi_norm)
    )
    return score


def score_hybrid_anti_chunk(
    capture_store,
    chunk_len: int,
    dims: List[int] = None,
    phi_layer: int = 14,
    alpha: float = 0.5,
    grounding_layers: List[int] = None,
) -> np.ndarray:
    """
    Anti-hybrid: high attn + high phi = attention sink.
    score = alpha * attn_norm + (1-alpha) * phi_norm

    Evict highest scorers to remove attention sinks.
    """
    if grounding_layers is None:
        grounding_layers = [18, 22, 24]

    phi = score_phi_chunk(
        capture_store.hidden_states, 0, chunk_len,
        dims=dims, layer=phi_layer,
    )
    attn = score_attn_chunk(
        capture_store.attn_weights, chunk_len,
        grounding_layers=grounding_layers,
    )

    phi_norm = _normalize_minmax(phi)
    attn_norm = _normalize_minmax(attn)

    # High attn + high phi = attention sink
    score = (
        alpha * attn_norm
        + (1 - alpha) * phi_norm
    )
    return score


# ── KV cache manipulation ──────────────────────────────

def evict_kv_by_indices(cache, evict_indices):
    """Remove specific cache entries by absolute index."""
    if len(evict_indices) == 0:
        return
    evict_set = set(int(i) for i in evict_indices)
    total = cache[0].offset
    keep = np.array(
        [i for i in range(total) if i not in evict_set],
        dtype=np.int32,
    )
    kmx = mx.array(keep)
    for lc in cache:
        if lc.keys is None:
            continue
        o = lc.offset
        lc.keys = lc.keys[:, :, :o, :][:, :, kmx, :]
        lc.values = lc.values[:, :, :o, :][:, :, kmx, :]
        lc.offset = len(keep)
    mx.eval([c.state for c in cache])


# ── Streaming prefill + eviction ────────────────────────

def streaming_prefill(
    model, proc, ids, pv, mask, extra,
    vis_start, vis_end, budget_frac,
    strategy, sink_dims, phi_layer=14,
    n_chunks=None,
):
    """
    Process input in chunks with streaming eviction.

    Returns (cache, final_logits) ready for generation.
    """
    from mlx_vlm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model.language_model)
    n_vis = vis_end - vis_start
    vis_budget = max(1, int(n_vis * budget_frac))

    # Get full embeddings + position_ids
    eo = model.get_input_embeddings(ids, pv, mask=mask, **extra)
    embeds = eo.inputs_embeds  # [1, seq_len, D]
    full_pos = model.language_model._position_ids  # [3, 1, seq]

    seq_len = embeds.shape[1]

    # Determine chunk boundaries for visual tokens
    if n_chunks is None:
        # Default: ~50 tokens per chunk
        n_chunks = max(1, n_vis // 50)
    chunk_size = n_vis // n_chunks
    remainder = n_vis - chunk_size * n_chunks

    vis_chunks = []
    pos = vis_start
    for ci in range(n_chunks):
        sz = chunk_size + (1 if ci < remainder else 0)
        vis_chunks.append((pos, pos + sz))
        pos += sz

    # Track which absolute cache positions are visual tokens
    # and their scores
    vis_cache_positions = []  # (cache_pos, phi_score)

    # ── Step 1: Process system tokens [0, vis_start) ────
    if vis_start > 0:
        sys_embeds = embeds[:, :vis_start, :]
        sys_pos = full_pos[:, :, :vis_start]
        out = model.language_model(
            None,
            inputs_embeds=sys_embeds,
            cache=cache,
            position_ids=sys_pos,
        )
        mx.eval([c.state for c in cache])

    # ── Step 2: Process visual chunks with eviction ─────
    for ci, (cs, ce) in enumerate(vis_chunks):
        chunk_len = ce - cs

        # Capture hidden states for φ scoring
        CaptureStore.enable()

        chunk_embeds = embeds[:, cs:ce, :]
        chunk_pos = full_pos[:, :, cs:ce]
        out = model.language_model(
            None,
            inputs_embeds=chunk_embeds,
            cache=cache,
            position_ids=chunk_pos,
        )
        mx.eval([c.state for c in cache])

        CaptureStore.disable()

        # Score this chunk's tokens
        hs = CaptureStore.hidden_states
        if hs and strategy in ("sink_aware", "anti_sink"):
            phi_scores = score_phi_chunk(
                hs, cs, chunk_len, sink_dims, phi_layer,
            )
        elif hs and strategy == "anti_sink_multi":
            phi_scores = score_phi_multilayer_chunk(
                hs, cs, chunk_len, sink_dims,
            )
        elif hs and strategy == "hybrid":
            phi_scores = score_hybrid_chunk(
                CaptureStore, chunk_len,
                dims=sink_dims, phi_layer=phi_layer,
            )
        elif hs and strategy == "hybrid_anti":
            phi_scores = score_hybrid_anti_chunk(
                CaptureStore, chunk_len,
                dims=sink_dims, phi_layer=phi_layer,
            )
        else:
            phi_scores = np.zeros(chunk_len)

        # Record visual tokens in cache
        # They occupy positions [cache_offset - chunk_len,
        #                        cache_offset)
        cache_offset = cache[0].offset
        for j in range(chunk_len):
            vis_cache_positions.append({
                "cache_pos": cache_offset - chunk_len + j,
                "seq_pos": cs + j,
                "phi": float(phi_scores[j]),
                "chunk": ci,
            })

        CaptureStore.reset()

        # ── Evict if over budget ────────────────────────
        n_vis_in_cache = len(vis_cache_positions)
        if n_vis_in_cache > vis_budget:
            n_evict = n_vis_in_cache - vis_budget
            evict_indices = _select_eviction(
                vis_cache_positions, n_evict, strategy,
            )
            evict_kv_by_indices(cache, evict_indices)

            # Update vis_cache_positions: remove evicted,
            # reindex remaining
            evict_set = set(evict_indices)
            old_positions = vis_cache_positions
            vis_cache_positions = []
            # Build mapping from old cache pos to new
            all_old = sorted(
                set(range(cache[0].offset + n_evict))
                - evict_set
            )
            # Actually, after eviction, positions are
            # compacted. We need to track correctly.
            remaining = [
                vp for vp in old_positions
                if vp["cache_pos"] not in evict_set
            ]
            # Re-number cache positions after compaction
            # The gather operation compacts entries
            old_to_new = {}
            new_idx = 0
            total_before = cache[0].offset + n_evict
            for i in range(total_before):
                if i not in evict_set:
                    old_to_new[i] = new_idx
                    new_idx += 1
            for vp in remaining:
                vp["cache_pos"] = old_to_new[vp["cache_pos"]]
            vis_cache_positions = remaining

    # ── Step 3: Process text tokens [vis_end, seq_len) ──
    if vis_end < seq_len:
        text_embeds = embeds[:, vis_end:, :]
        text_pos = full_pos[:, :, vis_end:]
        out = model.language_model(
            None,
            inputs_embeds=text_embeds,
            cache=cache,
            position_ids=text_pos,
        )
        mx.eval([c.state for c in cache])

    logits = out.logits[:, -1, :]
    return cache, logits, vis_cache_positions


def _select_eviction(vis_positions, n_evict, strategy):
    """Select which visual cache positions to evict."""
    if strategy == "sink_aware":
        # Keep HIGH phi (sinks) → evict LOW phi
        sorted_by_phi = sorted(
            vis_positions, key=lambda x: x["phi"],
        )
        return [vp["cache_pos"] for vp in sorted_by_phi[:n_evict]]

    elif strategy == "anti_sink":
        # Keep LOW phi (non-sinks) → evict HIGH phi
        sorted_by_phi = sorted(
            vis_positions, key=lambda x: -x["phi"],
        )
        return [vp["cache_pos"] for vp in sorted_by_phi[:n_evict]]

    elif strategy == "anti_sink_multi":
        # Like anti_sink but uses multi-layer φ scores.
        # Expects "phi" field already set via multi-layer
        # scoring in streaming_prefill.
        sorted_by_phi = sorted(
            vis_positions, key=lambda x: -x["phi"],
        )
        return [vp["cache_pos"] for vp in sorted_by_phi[:n_evict]]

    elif strategy == "uniform":
        # FIFO: evict oldest visual tokens first
        sorted_by_pos = sorted(
            vis_positions, key=lambda x: x["seq_pos"],
        )
        return [vp["cache_pos"] for vp in sorted_by_pos[:n_evict]]

    elif strategy == "random":
        rng = np.random.RandomState(42)
        chosen = rng.choice(
            len(vis_positions), n_evict, replace=False,
        )
        return [vis_positions[i]["cache_pos"] for i in chosen]

    elif strategy == "recency":
        # Keep most recent, evict oldest
        sorted_by_chunk = sorted(
            vis_positions, key=lambda x: x["chunk"],
        )
        return [
            vp["cache_pos"] for vp in sorted_by_chunk[:n_evict]
        ]

    elif strategy == "hybrid":
        # phi field holds hybrid score (high = keep)
        # → evict LOWEST scorers
        sorted_by_score = sorted(
            vis_positions, key=lambda x: x["phi"],
        )
        return [
            vp["cache_pos"] for vp in sorted_by_score[:n_evict]
        ]

    elif strategy == "hybrid_anti":
        # phi field holds hybrid_anti score
        # (high = attention sink) → evict HIGHEST scorers
        sorted_by_score = sorted(
            vis_positions, key=lambda x: -x["phi"],
        )
        return [
            vp["cache_pos"] for vp in sorted_by_score[:n_evict]
        ]

    elif strategy == "streamingllm":
        # StreamingLLM baseline: keep first N visual tokens
        # (attention sinks) + most recent tokens to fill
        # budget. Evict everything in between.
        n_keep = len(vis_positions) - n_evict
        n_sink = min(4, n_keep)
        n_recent = n_keep - n_sink

        sorted_by_seq = sorted(
            vis_positions, key=lambda x: x["seq_pos"],
        )
        sink_set = set(
            vp["cache_pos"] for vp in sorted_by_seq[:n_sink]
        )
        recent_set = set(
            vp["cache_pos"]
            for vp in sorted_by_seq[-n_recent:]
        ) if n_recent > 0 else set()

        protected = sink_set | recent_set
        return [
            vp["cache_pos"] for vp in vis_positions
            if vp["cache_pos"] not in protected
        ]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ── Generation from cache ──────────────────────────────

def generate_from_cache(model, proc, cache, logits,
                        max_tok=60):
    """Autoregressive generation from pre-filled cache."""
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


# ── Full pipeline (no eviction) for baseline ────────────

def full_prefill_generate(
    model, proc, ids, pv, mask, extra, max_tok=60,
):
    """Standard full prefill + generate (no eviction)."""
    from mlx_vlm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model.language_model)
    eo = model.get_input_embeddings(ids, pv, mask=mask, **extra)
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
    return generate_from_cache(model, proc, cache, logits,
                               max_tok)


def extract_answer(text):
    """Extract answer letter from model output."""
    text = text.strip()
    m = re.search(r'\b([A-D])\b', text)
    if m:
        return m.group(1)
    m = re.search(r'[Aa]nswer[:\s]*([A-D])', text)
    if m:
        return m.group(1)
    return text[:1].upper() if text else "?"


# ── Main ───────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-VL-7B-Instruct-8bit",
    )
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument(
        "--output-dir", default="sink_eviction_output_v9",
    )
    ap.add_argument(
        "--n-chunks", type=int, default=5,
        help="Number of visual token chunks (simulated frames)",
    )
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    budgets = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0]
    strategies = [
        "sink_aware", "anti_sink", "anti_sink_multi",
        "hybrid", "hybrid_anti",
        "uniform", "recency", "random",
        "streamingllm",
    ]
    sink_dims = [458, 2570]

    # Monkey-patch transformers bug: video_processor_class_from_name
    # crashes when VIDEO_PROCESSOR_MAPPING is None (no PyTorch).
    try:
        import transformers.models.auto.video_processing_auto as _vpa
        _orig = _vpa.video_processor_class_from_name
        def _safe_video_proc(class_name):
            try:
                return _orig(class_name)
            except TypeError:
                return None
        _vpa.video_processor_class_from_name = _safe_video_proc
    except Exception:
        pass

    from mlx_vlm import load
    from mlx_vlm.video_generate import process_vision_info

    print(f"Loading {args.model}...")
    model, processor = load(args.model)
    patch_model(model)
    mcfg = model.config

    def prepare(video, question, choices):
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
            return_tensors="np",
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

    # ── Check available videos ──────────────────────────
    existing = []
    for item in BENCHMARK:
        if os.path.exists(item["video"]):
            existing.append(item)
        else:
            print(f"  SKIP: {item['video']} not found")

    print(f"\n{'='*60}")
    print(f"v9: STREAMING EVICTION — MCQ ACCURACY")
    print(f"  {len(existing)} questions, "
          f"{len(budgets)} budgets, "
          f"{len(strategies)} strategies")
    print(f"  {args.n_chunks} visual chunks (simulated frames)")
    print(f"{'='*60}")

    # ── Baseline (no eviction) ──────────────────────────
    print("\n--- Baseline (full prefill, no eviction) ---")
    bl_correct = 0
    bl_total = 0
    for qi, item in enumerate(existing):
        ids, pv, mask, extra = prepare(
            item["video"], item["question"], item["choices"],
        )
        out = full_prefill_generate(
            model, processor, ids, pv, mask, extra,
            max_tok=args.max_tokens,
        )
        pred = extract_answer(out)
        correct = (pred == item["answer"])
        bl_correct += int(correct)
        bl_total += 1
        if qi < 3:
            print(f"  Q{qi+1}: '{out[:40]}' → {pred} "
                  f"{'ok' if correct else 'WRONG'} "
                  f"(gt={item['answer']})")
    bl_acc = bl_correct / bl_total if bl_total else 0
    print(f"  Baseline accuracy: {bl_acc:.1%} "
          f"({bl_correct}/{bl_total})")

    # ── Streaming eviction evaluation ───────────────────
    results = {
        sn: {b: [] for b in budgets}
        for sn in strategies
    }
    details = []

    for qi, item in enumerate(existing):
        print(f"\nQ{qi+1}: {item['question'][:50]}... "
              f"(gt={item['answer']})")

        ids, pv, mask, extra = prepare(
            item["video"], item["question"], item["choices"],
        )
        s, e = find_image_token_range(ids, mcfg)
        nv = e - s
        if qi == 0:
            print(f"  {nv} visual tokens, "
                  f"{args.n_chunks} chunks")

        for budget in budgets:
            for sn in strategies:
                if budget >= 1.0:
                    # No eviction at 100% — same as baseline
                    results[sn][budget].append(None)
                    continue

                cache, logits, vcp = streaming_prefill(
                    model, processor, ids, pv, mask, extra,
                    s, e, budget, sn, sink_dims,
                    n_chunks=args.n_chunks,
                )
                out = generate_from_cache(
                    model, processor, cache, logits,
                    max_tok=args.max_tokens,
                )
                pred = extract_answer(out)
                correct = (pred == item["answer"])
                results[sn][budget].append(correct)

                if budget == 0.40:
                    print(f"  {sn:>12} b=40%: "
                          f"'{out[:35]}' → {pred} "
                          f"{'ok' if correct else 'X'}")

                details.append({
                    "qi": qi,
                    "question": item["question"],
                    "gt": item["answer"],
                    "strategy": sn,
                    "budget": budget,
                    "pred": pred,
                    "output": out[:80],
                    "correct": correct,
                    "n_vis_kept": len(vcp),
                })

    # ── Aggregate results ───────────────────────────────
    print(f"\n{'='*60}")
    print("STREAMING EVICTION: MCQ ACCURACY")
    print(f"{'='*60}")
    print(f"\nBaseline (no eviction): {bl_acc:.1%}")

    acc_table = {}
    print(f"\n{'Budget':<8}", end="")
    for sn in strategies:
        print(f"{sn:<14}", end="")
    print()
    print("-" * (8 + 14 * len(strategies)))

    for b in budgets:
        if b >= 1.0:
            continue
        print(f"{b:<8.0%}", end="")
        for sn in strategies:
            valid = [r for r in results[sn][b]
                     if r is not None]
            if valid:
                acc = sum(valid) / len(valid)
                acc_table.setdefault(sn, {})[b] = acc
                print(f"{acc:<14.1%}", end="")
            else:
                print(f"{'N/A':<14}", end="")
        print()

    # ── Key comparison: sink_aware vs anti_sink ─────────
    print(f"\n{'='*60}")
    print("KEY COMPARISON: sink_aware vs anti_sink")
    print(f"{'='*60}")
    for b in [0.20, 0.40, 0.60]:
        sa = acc_table.get("sink_aware", {}).get(b, None)
        aa = acc_table.get("anti_sink", {}).get(b, None)
        uf = acc_table.get("uniform", {}).get(b, None)
        if sa is not None and aa is not None:
            diff = sa - aa
            winner = (
                "SINK_AWARE wins"
                if diff > 0.05
                else "ANTI_SINK wins"
                if diff < -0.05
                else "TIE"
            )
            print(f"  {b:.0%}: sink={sa:.1%}, "
                  f"anti={aa:.1%}, uniform={uf:.1%} "
                  f"→ {winner} (Δ={diff:+.1%})")

    # ── Plot ────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "sink_aware": "#E91E63",
        "anti_sink": "#2196F3",
        "anti_sink_multi": "#2E7D32",
        "hybrid": "#9C27B0",
        "hybrid_anti": "#00BCD4",
        "uniform": "#795548",
        "recency": "#4CAF50",
        "random": "#9E9E9E",
        "streamingllm": "#FF9800",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_budgets = [b for b in budgets if b < 1.0]

    for sn in strategies:
        if sn in acc_table:
            vals = [acc_table[sn].get(b) for b in plot_budgets]
            valid_b = [b for b, v in zip(plot_budgets, vals)
                       if v is not None]
            valid_v = [v for v in vals if v is not None]
            lw = 2.5 if sn in (
                "sink_aware", "anti_sink",
                "hybrid", "hybrid_anti",
            ) else 1.5
            ls = "-" if sn != "random" else "--"
            ax.plot(
                valid_b, valid_v, f"{ls}o",
                color=colors[sn], label=sn,
                linewidth=lw, markersize=6,
            )

    ax.axhline(y=bl_acc, color="black", linestyle="--",
               alpha=0.5, label=f"baseline ({bl_acc:.0%})")
    ax.set_xlabel("Budget (fraction of visual tokens kept)")
    ax.set_ylabel("MCQ Accuracy")
    ax.set_title(
        "v9: STREAMING Eviction — MCQ Accuracy\n"
        f"Eviction during prefill ({args.n_chunks} chunks), "
        f"{len(existing)} questions"
    )
    ax.legend()
    ax.set_xlim(0.15, 0.85)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fp = os.path.join(args.output_dir, "v9_streaming.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot → {fp}")

    # ── Save JSON ───────────────────────────────────────
    jp = os.path.join(args.output_dir, "results.json")
    with open(jp, "w") as f:
        json.dump({
            "experiment": "v9_streaming_eviction",
            "baseline_acc": bl_acc,
            "n_questions": len(existing),
            "n_chunks": args.n_chunks,
            "budgets": budgets,
            "strategies": strategies,
            "accuracy": {
                sn: {
                    str(b): acc_table.get(sn, {}).get(b)
                    for b in plot_budgets
                }
                for sn in strategies
            },
            "details": details,
        }, f, indent=2)
    print(f"JSON → {jp}")


if __name__ == "__main__":
    main()
