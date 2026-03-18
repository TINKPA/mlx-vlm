"""
POC v5: True KV Cache Eviction — the main experiment.

After prefill, we REMOVE entries from the KV cache by
slicing out evicted positions. This reduces cache size
and is exactly what streaming video VLMs need to do.

Compares eviction strategies:
  1. Sink-aware: keep sink tokens, evict non-sink
  2. Uniform: evict oldest visual tokens first
  3. Random: random eviction (control)
  4. Reverse-sink: keep non-sink, evict sink

At varying budget levels (% of visual tokens kept).

Usage:
  uv run --with "transformers<4.52" --with torch \
    --with matplotlib \
    python sink_eviction_poc_v5_true_evict.py \
      --video test_video.mp4 \
      --output-dir sink_eviction_output_v5
"""

import argparse
import os
import types
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sink_detect import (
    CaptureStore,
    find_image_token_range,
    patch_model,
    rmsnorm,
)


# ── Sink detection ──────────────────────────────────

def detect_sinks(
    hidden_states, sink_dims, img_start, img_end,
    tau=20.0, layer=14,
):
    """Returns (phi_values, sink_mask) for visual range."""
    hs = hidden_states[layer][0]
    rms = np.abs(rmsnorm(hs))
    phi = np.max(
        np.stack(
            [rms[:, d] for d in sink_dims], axis=-1
        ),
        axis=-1,
    )
    vis_phi = phi[img_start:img_end]
    sink_mask = vis_phi > tau
    return vis_phi, sink_mask


def token_overlap(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ── True KV Cache Eviction ──────────────────────────

def evict_kv_cache(
    prompt_cache: list,
    keep_indices: np.ndarray,
):
    """
    Remove entries from KV cache, keeping only
    specified positions. This truly reduces cache size.

    keep_indices: absolute sequence positions to retain
                  (sorted ascending)
    """
    keep = np.sort(keep_indices)
    keep_mx = mx.array(keep)

    for layer_cache in prompt_cache:
        # keys/values: [B, n_heads, seq_len, head_dim]
        k = layer_cache.keys
        v = layer_cache.values

        if k is None:
            continue

        # Slice to actual offset (cache may be padded)
        offset = layer_cache.offset
        k_actual = k[:, :, :offset, :]
        v_actual = v[:, :, :offset, :]

        # Gather kept positions
        k_new = k_actual[:, :, keep_mx, :]
        v_new = v_actual[:, :, keep_mx, :]

        new_len = len(keep)

        # Replace cache contents
        # Allocate new cache arrays
        layer_cache.keys = k_new
        layer_cache.values = v_new
        layer_cache.offset = new_len

    mx.eval([c.state for c in prompt_cache])


# ── Custom generate with eviction ───────────────────

def generate_with_eviction(
    model, processor, prompt_text,
    input_ids, pixel_values, mask,
    evict_indices: Optional[np.ndarray],
    max_tokens: int = 100,
    temperature: float = 0.0,
    **extra_kwargs,
) -> str:
    """
    Custom generate that does prefill → evict → decode.

    evict_indices: absolute positions to REMOVE from
    KV cache after prefill. If None, no eviction.
    """
    from mlx_vlm.models.cache import make_prompt_cache

    # Create fresh KV cache
    prompt_cache = make_prompt_cache(
        model.language_model,
    )

    # Get embeddings
    embedding_output = model.get_input_embeddings(
        input_ids, pixel_values,
        mask=mask, **extra_kwargs,
    )
    inputs_embeds = embedding_output.inputs_embeds
    fwd_kwargs = {
        k: v
        for k, v in embedding_output.to_dict().items()
        if k != "inputs_embeds" and v is not None
    }

    # ── Prefill ──
    outputs = model.language_model(
        input_ids,
        inputs_embeds=inputs_embeds,
        cache=prompt_cache,
        **fwd_kwargs,
    )
    mx.eval([c.state for c in prompt_cache])

    logits = outputs.logits[:, -1, :]

    # ── Evict from KV cache ──
    if evict_indices is not None and len(
        evict_indices
    ) > 0:
        seq_len = prompt_cache[0].offset
        all_positions = np.arange(seq_len)
        evict_set = set(evict_indices.tolist())
        keep = np.array(
            [i for i in all_positions
             if i not in evict_set],
            dtype=np.int32,
        )
        evict_kv_cache(prompt_cache, keep)

    # ── Decode ──
    if temperature == 0:
        y = mx.argmax(logits, axis=-1)
    else:
        y = mx.random.categorical(
            logits / temperature
        )

    tokens = [y.item()]
    fwd_kwargs_decode = {}

    for _ in range(max_tokens - 1):
        outputs = model.language_model(
            y[None],
            cache=prompt_cache,
            **fwd_kwargs_decode,
        )
        logits = outputs.logits[:, -1, :]

        if temperature == 0:
            y = mx.argmax(logits, axis=-1)
        else:
            y = mx.random.categorical(
                logits / temperature
            )

        tok = y.item()
        tokens.append(tok)

        # Stop on EOS
        if hasattr(processor, 'eos_token_id'):
            eos = processor.eos_token_id
        elif hasattr(processor, 'tokenizer'):
            eos = processor.tokenizer.eos_token_id
        else:
            eos = None

        if eos is not None:
            if isinstance(eos, list):
                if tok in eos:
                    break
            elif tok == eos:
                break

    text = processor.decode(
        tokens, skip_special_tokens=True,
    )
    return text


# ── Eviction strategies ─────────────────────────────

def strategy_sink_aware(
    vis_phi, sink_mask, img_start, img_end, budget,
):
    """
    Keep top-budget tokens by phi (strongest sinks first).
    """
    n_vis = img_end - img_start
    n_keep = int(n_vis * budget)
    if n_keep >= n_vis:
        return np.array([], dtype=np.int32)

    # Rank by phi descending
    ranked = np.argsort(vis_phi)[::-1]
    evict_local = ranked[n_keep:]
    return evict_local + img_start


def strategy_reverse_sink(
    vis_phi, sink_mask, img_start, img_end, budget,
):
    """
    Keep top-budget tokens by LOWEST phi (non-sinks first).
    """
    n_vis = img_end - img_start
    n_keep = int(n_vis * budget)
    if n_keep >= n_vis:
        return np.array([], dtype=np.int32)

    # Rank by phi ascending (keep non-sinks)
    ranked = np.argsort(vis_phi)
    evict_local = ranked[n_keep:]
    return evict_local + img_start


def strategy_uniform_oldest(
    vis_phi, sink_mask, img_start, img_end, budget,
):
    """
    Keep the NEWEST visual tokens (evict oldest first).
    This is what StreamingVLM does.
    """
    n_vis = img_end - img_start
    n_keep = int(n_vis * budget)
    if n_keep >= n_vis:
        return np.array([], dtype=np.int32)

    # Evict first (oldest) tokens
    n_evict = n_vis - n_keep
    evict_local = np.arange(n_evict)
    return evict_local + img_start


def strategy_random(
    vis_phi, sink_mask, img_start, img_end, budget,
    seed=42,
):
    """Random eviction."""
    n_vis = img_end - img_start
    n_keep = int(n_vis * budget)
    if n_keep >= n_vis:
        return np.array([], dtype=np.int32)

    rng = np.random.RandomState(seed)
    all_local = np.arange(n_vis)
    evict_local = rng.choice(
        all_local, size=n_vis - n_keep, replace=False,
    )
    return evict_local + img_start


STRATEGIES = {
    "sink_aware": strategy_sink_aware,
    "reverse_sink": strategy_reverse_sink,
    "uniform_oldest": strategy_uniform_oldest,
    "random": strategy_random,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=(
            "mlx-community/Qwen2.5-VL-7B-Instruct-8bit"
        ),
    )
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--prompt",
        default="Describe the butterfly in this video.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=100,
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        default="sink_eviction_output_v5",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sink_dims = [458, 2570]
    detect_layer = 14
    budgets = [0.08, 0.20, 0.40, 0.60, 0.80, 1.0]

    # ── Load model ──
    from mlx_vlm import load
    from mlx_vlm.utils import load_config
    from mlx_vlm.video_generate import process_vision_info

    print(f"Loading {args.model}...")
    model, processor = load(args.model)
    config = load_config(args.model)
    patch_model(model)

    # ── Prepare inputs ──
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": args.video,
                    "max_pixels": 224 * 224,
                    "fps": args.fps,
                },
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True,
    )
    img_inputs, vid_inputs, _ = (
        process_vision_info(messages, True)
    )
    inputs = processor(
        text=[text], images=img_inputs,
        videos=vid_inputs, padding=True,
        return_tensors="pt",
    )

    input_ids = mx.array(inputs["input_ids"])
    pv = inputs.get(
        "pixel_values_videos",
        inputs.get("pixel_values", None),
    )
    pixel_values = mx.array(pv)
    mask_input = mx.array(inputs["attention_mask"])
    seq_len = input_ids.shape[1]

    extra_kwargs = {}
    for key in ["video_grid_thw", "image_grid_thw"]:
        if inputs.get(key) is not None:
            extra_kwargs[key] = mx.array(inputs[key])

    model_config = model.config
    img_start, img_end = find_image_token_range(
        input_ids, model_config,
    )
    n_vis = img_end - img_start
    print(f"Seq len: {seq_len}, "
          f"Visual: [{img_start},{img_end}) = {n_vis}")

    # ══════════════════════════════════════════════════
    # BASELINE: full generation + sink detection
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("BASELINE + sink detection")
    print("=" * 60)

    CaptureStore.enable()
    baseline = generate_with_eviction(
        model, processor, text,
        input_ids, pixel_values, mask_input,
        evict_indices=None,
        max_tokens=args.max_tokens,
        **extra_kwargs,
    )
    CaptureStore.disable()
    print(f"  {baseline}")

    hs_np = [
        np.array(h) for h in CaptureStore.hidden_states
    ]
    vis_phi, sink_mask = detect_sinks(
        hs_np, sink_dims, img_start, img_end,
        tau=20.0, layer=detect_layer,
    )
    n_sink = int(sink_mask.sum())
    n_nonsink = n_vis - n_sink
    print(f"\nSink: {n_sink} ({100*n_sink/n_vis:.0f}%), "
          f"Non-sink: {n_nonsink} "
          f"({100*n_nonsink/n_vis:.0f}%)")

    # ══════════════════════════════════════════════════
    # MAIN EXPERIMENT: strategies x budgets
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("MAIN EXPERIMENT: True KV Cache Eviction")
    print("=" * 60)

    results = {}
    for strat_name, strat_fn in STRATEGIES.items():
        results[strat_name] = []
        for budget in budgets:
            n_keep = int(n_vis * budget)
            n_evict = n_vis - n_keep

            if n_evict == 0:
                out = baseline
            else:
                evict_idx = strat_fn(
                    vis_phi, sink_mask,
                    img_start, img_end, budget,
                )
                out = generate_with_eviction(
                    model, processor, text,
                    input_ids, pixel_values, mask_input,
                    evict_indices=evict_idx,
                    max_tokens=args.max_tokens,
                    **extra_kwargs,
                )

            sim = token_overlap(baseline, out)
            coherent = len(set(out.split())) > 5
            results[strat_name].append({
                "budget": budget,
                "n_keep": n_keep,
                "n_evict": n_evict,
                "output": out,
                "sim": sim,
                "coherent": coherent,
            })

            status = "OK" if coherent else "BROKEN"
            print(f"  {strat_name:<16} "
                  f"keep={n_keep:>3}/{n_vis} "
                  f"({budget:.0%}) "
                  f"sim={sim:.2f} [{status}] "
                  f"{out[:50]}")

    # ══════════════════════════════════════════════════
    # PLOT
    # ══════════════════════════════════════════════════
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {
        "sink_aware": "#2196F3",
        "reverse_sink": "#F44336",
        "uniform_oldest": "#FF9800",
        "random": "#9E9E9E",
    }
    labels = {
        "sink_aware": "Sink-Aware (ours)",
        "reverse_sink": "Reverse-Sink",
        "uniform_oldest": "Uniform Oldest",
        "random": "Random",
    }

    # Plot 1: Similarity vs budget
    ax = axes[0]
    for strat_name in STRATEGIES:
        bs = [r["budget"] for r in results[strat_name]]
        ss = [r["sim"] for r in results[strat_name]]
        ax.plot(
            bs, ss, "-o",
            color=colors[strat_name],
            label=labels[strat_name],
            linewidth=2, markersize=6,
        )
    ax.set_xlabel(
        "Budget (fraction of visual tokens kept)",
        fontsize=11,
    )
    ax.set_ylabel(
        "Token Overlap with Baseline", fontsize=11,
    )
    ax.set_title(
        "KV Cache Eviction: Quality vs Budget",
        fontsize=13,
    )
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.axhline(y=0.5, color="gray", ls="--", alpha=0.3)

    # Plot 2: Coherence heatmap
    ax = axes[1]
    strat_names = list(STRATEGIES.keys())
    heatmap = np.zeros(
        (len(strat_names), len(budgets)),
    )
    for i, sn in enumerate(strat_names):
        for j, r in enumerate(results[sn]):
            heatmap[i, j] = r["sim"]

    im = ax.imshow(
        heatmap, cmap="RdYlGn", vmin=0, vmax=1,
        aspect="auto",
    )
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels(
        [f"{b:.0%}" for b in budgets], fontsize=9,
    )
    ax.set_yticks(range(len(strat_names)))
    ax.set_yticklabels(
        [labels[s] for s in strat_names], fontsize=9,
    )
    ax.set_xlabel("Budget", fontsize=11)
    ax.set_title(
        "Quality Heatmap\n"
        "(green=good, red=bad)",
        fontsize=13,
    )

    # Add text annotations
    for i in range(len(strat_names)):
        for j in range(len(budgets)):
            val = heatmap[i, j]
            coh = results[strat_names[i]][j]["coherent"]
            txt = f"{val:.2f}"
            if not coh:
                txt += "\nBROKEN"
            color = "white" if val < 0.4 else "black"
            ax.text(
                j, i, txt, ha="center", va="center",
                fontsize=8, color=color,
            )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    fig_path = os.path.join(
        args.output_dir, "eviction_comparison.png",
    )
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved -> {fig_path}")

    # ── Save text results ──
    txt_path = os.path.join(
        args.output_dir, "results.txt",
    )
    with open(txt_path, "w") as f:
        f.write("Sink Eviction v5 — "
                "True KV Cache Eviction\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Video: {args.video}\n")
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"Visual: {n_vis} "
                f"(sink={n_sink}, non={n_nonsink})\n")
        f.write(f"Detection: L{detect_layer}, tau=20\n")
        f.write(f"Method: True KV cache eviction "
                f"(entries removed)\n\n")

        f.write(f"Baseline:\n{baseline}\n\n")

        for strat_name in STRATEGIES:
            f.write(f"\n{'='*40}\n")
            f.write(f"Strategy: {labels[strat_name]}\n")
            f.write(f"{'='*40}\n")
            for r in results[strat_name]:
                f.write(
                    f"\n  Budget {r['budget']:.0%} "
                    f"(keep={r['n_keep']}, "
                    f"evict={r['n_evict']})\n"
                )
                f.write(f"  sim={r['sim']:.2f} "
                        f"{'OK' if r['coherent'] else 'BROKEN'}\n")
                f.write(f"  {r['output'][:120]}\n")

    print(f"Results saved -> {txt_path}")

    # ── Key findings ──
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    # Find minimum budget for each strategy to be OK
    for strat_name in STRATEGIES:
        for r in results[strat_name]:
            if r["coherent"] and r["sim"] > 0.3:
                print(
                    f"  {labels[strat_name]:<20} "
                    f"min viable budget: "
                    f"{r['budget']:.0%} "
                    f"(keep {r['n_keep']}/{n_vis}, "
                    f"sim={r['sim']:.2f})"
                )
                break
        else:
            print(
                f"  {labels[strat_name]:<20} "
                f"NEVER viable (all budgets broken)"
            )


if __name__ == "__main__":
    main()
