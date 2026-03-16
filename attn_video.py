"""
Per-frame attention overlay video.

For each sampled frame: run single-image inference with the same
prompt, extract attention from the best visual head, overlay the
spatial heatmap, and stitch into an output video.

Usage:
  uv run --with "transformers<4.52" --with torch \
    --with seaborn --with matplotlib \
    python attn_video.py \
      --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit \
      --video test_video.mp4 \
      --prompt "What is the main subject doing?" \
      --fps 2.0 \
      --output attn_overlay.mp4
"""

import argparse
import os
import sys
import time

import cv2
import mlx.core as mx
import numpy as np
from PIL import Image

from attn_viz import AttentionCapture, patch_model


def extract_frames(video_path, fps):
    """Extract frames from video at target fps."""
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / video_fps

    step = max(1, int(video_fps / fps))
    frames = []
    indices = []
    idx = 0
    while idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        indices.append(idx)
        idx += step
    cap.release()

    print(f"Video: {total} frames @ {video_fps:.1f}fps, "
          f"duration={duration:.1f}s")
    print(f"Sampled {len(frames)} frames "
          f"(step={step}, effective {fps}fps)")
    return frames, indices, video_fps


def run_single_frame(model, processor, frame_bgr, prompt,
                     tmp_path="/tmp/_attn_frame.jpg"):
    """Run inference on one frame, return attention + metadata."""
    from mlx_vlm import generate
    from mlx_vlm.video_generate import process_vision_info

    # Save frame temporarily
    cv2.imwrite(tmp_path, frame_bgr)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": tmp_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs, _ = process_vision_info(
        messages, True,
    )
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )

    input_ids = mx.array(inputs["input_ids"])
    pixel_values = inputs.get(
        "pixel_values", inputs.get("pixel_values_videos"),
    )
    pixel_values = mx.array(pixel_values)
    mask_input = mx.array(inputs["attention_mask"])

    kwargs = {}
    grid_thw = None
    if inputs.get("image_grid_thw") is not None:
        kwargs["image_grid_thw"] = mx.array(
            inputs["image_grid_thw"]
        )
        grid_thw = [
            int(x)
            for x in np.array(inputs["image_grid_thw"][0])
        ]

    # Find image token range
    ids_list = input_ids.flatten().tolist()
    img_id = model.config.image_token_id
    positions = [i for i, t in enumerate(ids_list) if t == img_id]
    if not positions:
        vid_id = model.config.video_token_id
        positions = [
            i for i, t in enumerate(ids_list) if t == vid_id
        ]
    img_start = positions[0]
    img_end = positions[-1] + 1

    AttentionCapture.enable()

    kwargs["input_ids"] = input_ids
    kwargs["pixel_values"] = pixel_values
    kwargs["mask"] = mask_input
    kwargs["temperature"] = 0.0
    kwargs["max_tokens"] = 1  # only need prefill attention

    output = generate(
        model, processor, prompt=text,
        verbose=False, **kwargs,
    )
    AttentionCapture.enabled = False

    attn_list = AttentionCapture.weights  # list of mx.array
    return attn_list, img_start, img_end, grid_thw


def compute_attention_rollout(attn_list, head_fusion="mean"):
    """
    Attention Rollout (Abnar & Zuidema, 2020).

    For each layer:
      1. Fuse heads (mean, max, or min)
      2. Add residual identity: A = 0.5*A + 0.5*I
      3. Re-normalize rows
      4. Multiply: rollout = A_0 @ A_1 @ ... @ A_L

    Returns rollout matrix [Q, K].
    """
    rollout = None
    for aw in attn_list:
        a = np.array(aw)[0]  # [n_heads, Q, K]

        # Fuse heads
        if head_fusion == "mean":
            a_fused = a.mean(axis=0)
        elif head_fusion == "max":
            a_fused = a.max(axis=0)
        elif head_fusion == "min":
            a_fused = a.min(axis=0)
        else:
            raise ValueError(f"Unknown fusion: {head_fusion}")

        # Add residual connection (identity)
        seq_len = a_fused.shape[0]
        I = np.eye(seq_len)
        a_hat = 0.5 * a_fused + 0.5 * I

        # Re-normalize rows to sum to 1
        a_hat = a_hat / a_hat.sum(axis=-1, keepdims=True)

        # Accumulate
        if rollout is None:
            rollout = a_hat
        else:
            rollout = rollout @ a_hat

    return rollout


def find_best_visual_head(attn_list, img_start, img_end):
    """Find (layer, head) with highest txt→img ratio."""
    best = (0, 0, 0.0)
    for li, aw in enumerate(attn_list):
        a = np.array(aw)[0]  # [n_heads, Q, K]
        seq_len = a.shape[-1]
        n_heads = a.shape[0]

        txt_mask = np.ones(seq_len, dtype=bool)
        txt_mask[img_start:img_end] = False
        img_mask = ~txt_mask

        for hi in range(n_heads):
            txt_rows = a[hi][txt_mask]
            if txt_rows.shape[0] == 0:
                continue
            ratio = txt_rows[:, img_mask].sum(axis=1).mean()
            if ratio > best[2]:
                best = (li, hi, ratio)
    return best


def get_spatial_heatmap(attn_list, layer, head,
                        img_start, img_end, grid_thw,
                        spatial_merge_size=2,
                        use_rollout=False,
                        head_fusion="mean"):
    """Extract spatial attention heatmap.

    If use_rollout=True, uses Attention Rollout across all layers
    instead of a single (layer, head).
    """
    if use_rollout:
        rollout = compute_attention_rollout(
            attn_list, head_fusion=head_fusion,
        )
        # Last token → image tokens
        last_idx = rollout.shape[0] - 1
        img_attn = rollout[last_idx, img_start:img_end]
    else:
        aw = np.array(attn_list[layer])[0]  # [n_heads, Q, K]
        head_attn = aw[head]  # [Q, K]
        last_idx = head_attn.shape[0] - 1
        img_attn = head_attn[last_idx, img_start:img_end]

    T, H, W = grid_thw
    llm_H = H // spatial_merge_size
    llm_W = W // spatial_merge_size

    expected = llm_H * llm_W
    actual = img_attn.shape[0]

    if actual >= expected:
        img_attn = img_attn[:expected]
    else:
        img_attn = np.pad(img_attn, (0, expected - actual))

    heatmap = img_attn.reshape(llm_H, llm_W)
    return heatmap


def overlay_heatmap(frame_bgr, heatmap, alpha=0.45):
    """Overlay attention heatmap on frame."""
    h, w = frame_bgr.shape[:2]

    # Normalize
    hm = heatmap.astype(np.float32)
    hm_min, hm_max = hm.min(), hm.max()
    if hm_max - hm_min > 1e-8:
        hm = (hm - hm_min) / (hm_max - hm_min)
    else:
        hm = np.zeros_like(hm)

    # Upsample to frame size
    hm_resized = cv2.resize(hm, (w, h), interpolation=cv2.INTER_CUBIC)
    hm_color = cv2.applyColorMap(
        (hm_resized * 255).astype(np.uint8), cv2.COLORMAP_JET,
    )

    overlay = cv2.addWeighted(frame_bgr, 1 - alpha, hm_color, alpha, 0)
    return overlay


def main():
    parser = argparse.ArgumentParser(
        description="Per-frame attention overlay video",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    )
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--prompt",
        default="What is the main subject doing?",
    )
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--output", default="attn_overlay.mp4")
    parser.add_argument(
        "--output-fps", type=float, default=4.0,
        help="FPS of the output video",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.45,
        help="Heatmap overlay opacity",
    )
    parser.add_argument(
        "--rollout", action="store_true",
        help="Use Attention Rollout instead of single head",
    )
    parser.add_argument(
        "--head-fusion", default="mean",
        choices=["mean", "max", "min"],
        help="Head fusion strategy for rollout",
    )
    args = parser.parse_args()

    # ── Load model ──
    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    print(f"Loading model: {args.model}")
    model, processor = load(args.model)
    config = load_config(args.model)
    patch_model(model)

    spatial_merge_size = model.config.vision_config.spatial_merge_size

    # ── Extract frames ──
    frames, indices, video_fps = extract_frames(
        args.video, args.fps,
    )

    # ── First pass: find best visual head ──
    if args.rollout:
        print(f"\nUsing Attention Rollout "
              f"(head_fusion={args.head_fusion})")
        best_layer, best_head, best_ratio = 0, 0, 0.0
        # Still need a calibration run to verify it works
        mid = len(frames) // 2
        attn_list, img_s, img_e, grid_thw = run_single_frame(
            model, processor, frames[mid], args.prompt,
        )
        n_layers = len(attn_list)
        print(f"Rollout across {n_layers} layers")
    else:
        print("\nCalibrating: finding best visual head "
              "from middle frame...")
        mid = len(frames) // 2
        attn_list, img_s, img_e, grid_thw = run_single_frame(
            model, processor, frames[mid], args.prompt,
        )
        best_layer, best_head, best_ratio = find_best_visual_head(
            attn_list, img_s, img_e,
        )
        print(f"Best visual head: Layer {best_layer}, "
              f"Head {best_head} (ratio={best_ratio:.3f})")

    # ── Per-frame processing ──
    print(f"\nProcessing {len(frames)} frames...")
    overlay_frames = []
    t0 = time.time()

    for i, frame in enumerate(frames):
        attn_list, img_s, img_e, grid_thw = run_single_frame(
            model, processor, frame, args.prompt,
        )

        heatmap = get_spatial_heatmap(
            attn_list, best_layer, best_head,
            img_s, img_e, grid_thw, spatial_merge_size,
            use_rollout=args.rollout,
            head_fusion=args.head_fusion,
        )

        overlay = overlay_heatmap(frame, heatmap, args.alpha)

        # Add text annotation
        t_sec = indices[i] / video_fps
        if args.rollout:
            mode_str = f"Rollout({args.head_fusion})"
        else:
            mode_str = f"L{best_layer} H{best_head}"
        label = (f"{mode_str} | "
                 f"t={t_sec:.1f}s | "
                 f"frame {i+1}/{len(frames)}")
        cv2.putText(
            overlay, label, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 2, cv2.LINE_AA,
        )

        overlay_frames.append(overlay)

        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(frames) - i - 1)
        print(f"  [{i+1}/{len(frames)}] "
              f"t={t_sec:.1f}s  "
              f"({elapsed:.0f}s elapsed, ~{eta:.0f}s left)")

        # Free memory
        del attn_list
        mx.clear_cache()

    # ── Write output video ──
    h, w = overlay_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        args.output, fourcc, args.output_fps, (w, h),
    )
    for frame in overlay_frames:
        out.write(frame)
    out.release()

    total_time = time.time() - t0
    print(f"\nDone! {len(overlay_frames)} frames → {args.output}")
    print(f"Output: {w}x{h} @ {args.output_fps}fps, "
          f"duration={len(overlay_frames)/args.output_fps:.1f}s")
    print(f"Total time: {total_time:.0f}s "
          f"({total_time/len(frames):.1f}s/frame)")


if __name__ == "__main__":
    main()
