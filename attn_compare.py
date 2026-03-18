"""
Compare attention patterns: single image vs video.
Produces side-by-side Stage 1 & Stage 2 comparisons.
"""

import cv2
import mlx.core as mx
import numpy as np
import os

from PIL import Image


def run_image_capture(model, processor, config, image_path, prompt):
    """Run model on a single image, return captured attentions + metadata."""
    from mlx_vlm import generate
    from mlx_vlm.video_generate import process_vision_info
    from attn_viz import AttentionCapture

    # Use the same processor path as video to get proper
    # image token expansion
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
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
            int(x) for x in np.array(inputs["image_grid_thw"][0])
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
    img_start, img_end = positions[0], positions[-1] + 1

    AttentionCapture.enable()

    kwargs["input_ids"] = input_ids
    kwargs["pixel_values"] = pixel_values
    kwargs["mask"] = mask_input
    kwargs["temperature"] = 0.0
    kwargs["max_tokens"] = 50

    output = generate(
        model, processor, prompt=text,
        verbose=False, **kwargs,
    )
    AttentionCapture.enabled = False

    attn_np = [np.array(aw) for aw in AttentionCapture.weights]
    seq_len = attn_np[0].shape[-1]

    return {
        "attn": attn_np,
        "img_start": img_start,
        "img_end": img_end,
        "seq_len": seq_len,
        "n_img_tokens": img_end - img_start,
        "grid_thw": grid_thw,
        "output": output.text,
    }


def run_video_capture(model, processor, video_path, prompt, fps=1.0):
    """Run model on video, return captured attentions + metadata."""
    from mlx_vlm import generate
    from mlx_vlm.video_generate import process_vision_info
    from attn_viz import AttentionCapture

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 224 * 224,
                    "fps": fps,
                },
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
        "pixel_values_videos", inputs.get("pixel_values"),
    )
    pixel_values = mx.array(pixel_values)
    mask_input = mx.array(inputs["attention_mask"])

    kwargs = {}
    if inputs.get("video_grid_thw") is not None:
        kwargs["video_grid_thw"] = mx.array(
            inputs["video_grid_thw"]
        )
        grid_thw = [
            int(x) for x in np.array(inputs["video_grid_thw"][0])
        ]
    elif inputs.get("image_grid_thw") is not None:
        kwargs["image_grid_thw"] = mx.array(
            inputs["image_grid_thw"]
        )
        grid_thw = [
            int(x) for x in np.array(inputs["image_grid_thw"][0])
        ]

    # Find image/video token range
    ids_list = input_ids.flatten().tolist()
    img_id = model.config.image_token_id
    vid_id = model.config.video_token_id
    token_id = img_id if img_id in ids_list else vid_id
    positions = [i for i, t in enumerate(ids_list) if t == token_id]
    img_start, img_end = positions[0], positions[-1] + 1

    AttentionCapture.enable()

    kwargs["video"] = [video_path]
    kwargs["input_ids"] = input_ids
    kwargs["pixel_values"] = pixel_values
    kwargs["mask"] = mask_input
    kwargs["temperature"] = 0.0
    kwargs["max_tokens"] = 50

    output = generate(
        model, processor, prompt=text,
        verbose=False, **kwargs,
    )
    AttentionCapture.enabled = False

    attn_np = [np.array(aw) for aw in AttentionCapture.weights]
    seq_len = attn_np[0].shape[-1]

    return {
        "attn": attn_np,
        "img_start": img_start,
        "img_end": img_end,
        "seq_len": seq_len,
        "n_img_tokens": img_end - img_start,
        "grid_thw": grid_thw,
        "output": output.text,
    }


def compute_modality_flows(attn_list, img_start, img_end):
    """Compute 4-column modality flow matrix [n_layers, 4]."""
    n_layers = len(attn_list)
    flows = np.zeros((n_layers, 4))

    for li, aw in enumerate(attn_list):
        a = aw[0].mean(axis=0)  # [Q, K], averaged over heads
        seq_len = a.shape[0]

        txt_mask = np.ones(seq_len, dtype=bool)
        txt_mask[img_start:img_end] = False
        img_mask = ~txt_mask

        def flow(q_mask, k_mask):
            rows = a[q_mask]
            if rows.shape[0] == 0:
                return 0.0
            return rows[:, k_mask].sum(axis=1).mean()

        flows[li, 0] = flow(txt_mask, txt_mask)
        flows[li, 1] = flow(txt_mask, img_mask)
        flows[li, 2] = flow(img_mask, txt_mask)
        flows[li, 3] = flow(img_mask, img_mask)

    return flows


def compute_head_ratios(attn_list, img_start, img_end):
    """Compute txt→img ratio for each (layer, head)."""
    seq_len = attn_list[0].shape[-1]
    txt_mask = np.ones(seq_len, dtype=bool)
    txt_mask[img_start:img_end] = False
    img_mask = ~txt_mask

    results = []
    for li, aw in enumerate(attn_list):
        a = aw[0]  # [n_heads, Q, K]
        n_heads = a.shape[0]
        for hi in range(n_heads):
            txt_rows = a[hi][txt_mask]
            if txt_rows.shape[0] == 0:
                continue
            ratio = txt_rows[:, img_mask].sum(axis=1).mean()
            results.append((li, hi, ratio))
    return results


def plot_comparison(img_result, vid_result, output_dir):
    """Generate side-by-side comparison plots."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)
    labels = ["txt→txt", "txt→img", "img→txt", "img→img"]

    img_flows = compute_modality_flows(
        img_result["attn"],
        img_result["img_start"],
        img_result["img_end"],
    )
    vid_flows = compute_modality_flows(
        vid_result["attn"],
        vid_result["img_start"],
        vid_result["img_end"],
    )

    n_layers = img_flows.shape[0]
    layer_labels = [f"L{i}" for i in range(n_layers)]

    # ── Stage 1 comparison: side-by-side heatmaps ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10))

    sns.heatmap(
        img_flows, annot=True, fmt=".2f",
        xticklabels=labels, yticklabels=layer_labels,
        cmap="YlOrRd", vmin=0, vmax=1, ax=ax1,
        annot_kws={"size": 7},
    )
    ax1.set_title(
        f"Single Image\n"
        f"({img_result['n_img_tokens']} img tokens, "
        f"seq={img_result['seq_len']})",
        fontsize=11,
    )
    ax1.set_ylabel("Layer")

    sns.heatmap(
        vid_flows, annot=True, fmt=".2f",
        xticklabels=labels, yticklabels=layer_labels,
        cmap="YlOrRd", vmin=0, vmax=1, ax=ax2,
        annot_kws={"size": 7},
    )
    ax2.set_title(
        f"Video (5 frames)\n"
        f"({vid_result['n_img_tokens']} img tokens, "
        f"seq={vid_result['seq_len']})",
        fontsize=11,
    )
    ax2.set_ylabel("")

    fig.suptitle(
        "Stage 1: Modality Flow — Image vs Video",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "compare_stage1.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Stage 1 comparison saved → {path}")

    # ── Difference heatmap (video - image) ──
    diff = vid_flows - img_flows
    fig, ax = plt.subplots(figsize=(6, 10))
    sns.heatmap(
        diff, annot=True, fmt="+.2f",
        xticklabels=labels, yticklabels=layer_labels,
        cmap="RdBu_r", center=0, vmin=-0.3, vmax=0.3, ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title(
        "Stage 1 Δ: Video − Image\n"
        "(red = video higher, blue = image higher)",
        fontsize=11,
    )
    ax.set_ylabel("Layer")
    plt.tight_layout()
    path = os.path.join(output_dir, "compare_stage1_diff.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Stage 1 diff saved → {path}")

    # ── Stage 2 comparison: overlaid scatter ──
    img_heads = compute_head_ratios(
        img_result["attn"],
        img_result["img_start"],
        img_result["img_end"],
    )
    vid_heads = compute_head_ratios(
        vid_result["attn"],
        vid_result["img_start"],
        vid_result["img_end"],
    )

    fig, ax = plt.subplots(figsize=(14, 6))

    img_layers = [h[0] for h in img_heads]
    img_ratios = [h[2] for h in img_heads]
    vid_layers = [h[0] for h in vid_heads]
    vid_ratios = [h[2] for h in vid_heads]

    ax.scatter(
        np.array(img_layers) - 0.15, img_ratios,
        c="steelblue", alpha=0.5, s=20,
        edgecolors="none", label="Single Image",
    )
    ax.scatter(
        np.array(vid_layers) + 0.15, vid_ratios,
        c="orangered", alpha=0.5, s=20,
        edgecolors="none", label="Video",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("txt→img Attention Ratio")
    ax.set_title(
        "Stage 2: Head Specialization — Image vs Video",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir, "compare_stage2.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Stage 2 comparison saved → {path}")

    # ── Summary stats ──
    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"{'':20s} {'Image':>12s} {'Video':>12s}")
    print("-" * 55)
    print(f"{'Seq length':20s} {img_result['seq_len']:12d} "
          f"{vid_result['seq_len']:12d}")
    print(f"{'Image/video tokens':20s} "
          f"{img_result['n_img_tokens']:12d} "
          f"{vid_result['n_img_tokens']:12d}")
    print(f"{'Avg txt→img (all L)':20s} "
          f"{img_flows[:, 1].mean():12.3f} "
          f"{vid_flows[:, 1].mean():12.3f}")
    print(f"{'Max txt→img (any L)':20s} "
          f"{img_flows[:, 1].max():12.3f} "
          f"{vid_flows[:, 1].max():12.3f}")
    print(f"{'Avg img→img (all L)':20s} "
          f"{img_flows[:, 3].mean():12.3f} "
          f"{vid_flows[:, 3].mean():12.3f}")

    img_top = max(img_ratios)
    vid_top = max(vid_ratios)
    print(f"{'Top head txt→img':20s} "
          f"{img_top:12.3f} {vid_top:12.3f}")
    print("=" * 55)


def main():
    from mlx_vlm import load
    from mlx_vlm.utils import load_config
    from attn_viz import patch_model

    model_path = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
    output_dir = "attn_compare_output"

    print(f"Loading model: {model_path}")
    model, processor = load(model_path)
    config = load_config(model_path)
    patch_model(model)

    # Extract a representative frame from the video
    cap = cv2.VideoCapture("test_video.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 125)
    ret, frame = cap.read()
    cap.release()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img.save("test_single_frame.jpg")
    print(f"Extracted middle frame: {img.size}")

    prompt = "Describe what is happening in detail."

    # ── Run single image ──
    print("\n" + "=" * 55)
    print("Running: SINGLE IMAGE")
    print("=" * 55)
    img_result = run_image_capture(
        model, processor, config,
        "test_single_frame.jpg", prompt,
    )
    print(f"  Output: {img_result['output'][:80]}...")
    print(f"  Seq len: {img_result['seq_len']}, "
          f"img tokens: {img_result['n_img_tokens']}")

    # ── Run video ──
    print("\n" + "=" * 55)
    print("Running: VIDEO (5 frames @ 1fps)")
    print("=" * 55)
    vid_result = run_video_capture(
        model, processor,
        "test_video.mp4", prompt, fps=1.0,
    )
    print(f"  Output: {vid_result['output'][:80]}...")
    print(f"  Seq len: {vid_result['seq_len']}, "
          f"img tokens: {vid_result['n_img_tokens']}")

    # ── Compare ──
    print("\n" + "=" * 55)
    print("Generating comparison plots...")
    print("=" * 55)
    plot_comparison(img_result, vid_result, output_dir)

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
