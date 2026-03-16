"""
Visualize hidden state activation spikes at sink dimensions.
Reproduces the paper's Fig 2 style: RMSNorm(h) across all dims.

Shows BOS token vs visual sink token vs visual non-sink token,
with vertical lines marking dims {458, 2570}.
"""

import argparse
import os
from typing import List

import mlx.core as mx
import numpy as np

from sink_detect import (
    CaptureStore,
    find_image_token_range,
    patch_model,
    rmsnorm,
)


def plot_dim_spikes(
    hs_np: List[np.ndarray],
    sink_dims: List[int],
    img_start: int,
    img_end: int,
    output_dir: str,
):
    import matplotlib.pyplot as plt

    n_layers = len(hs_np)
    hidden_dim = hs_np[0].shape[-1]
    sample_layers = [2, 5, 10, 15, 20, 25]
    sample_layers = [l for l in sample_layers if l < n_layers]

    fig, axes = plt.subplots(
        len(sample_layers), 3,
        figsize=(18, 3.5 * len(sample_layers)),
        sharey="row",
    )

    col_labels = ["BOS token", "Visual SINK token",
                   "Visual non-SINK token"]

    for row, li in enumerate(sample_layers):
        hs = hs_np[li][0]  # [seq, dim]
        rms = np.abs(rmsnorm(hs))

        # Compute phi to find sink / non-sink
        phi = np.max(
            np.stack([rms[:, d] for d in sink_dims], axis=-1),
            axis=-1,
        )
        vis_phi = phi[img_start:img_end]

        # Pick representative tokens
        bos_rms = rms[0]  # BOS

        # Sink: highest phi visual token
        sink_idx_local = np.argmax(vis_phi)
        sink_rms = rms[img_start + sink_idx_local]
        sink_phi = vis_phi[sink_idx_local]

        # Non-sink: lowest phi visual token
        nonsink_idx_local = np.argmin(vis_phi)
        nonsink_rms = rms[img_start + nonsink_idx_local]
        nonsink_phi = vis_phi[nonsink_idx_local]

        tokens = [
            (bos_rms, f"BOS (φ={phi[0]:.1f})", "red"),
            (sink_rms,
             f"Vis sink #{sink_idx_local} "
             f"(φ={sink_phi:.1f})", "orange"),
            (nonsink_rms,
             f"Vis non-sink #{nonsink_idx_local} "
             f"(φ={nonsink_phi:.1f})", "steelblue"),
        ]

        for col, (activation, label, color) in enumerate(
            tokens
        ):
            ax = axes[row, col]
            ax.plot(
                range(hidden_dim), activation,
                color=color, linewidth=0.3, alpha=0.8,
            )
            # Mark sink dims with vertical lines
            for d in sink_dims:
                val = activation[d]
                ax.axvline(
                    x=d, color="black", linestyle="--",
                    linewidth=1, alpha=0.5,
                )
                ax.plot(
                    d, val, "ko", markersize=6, zorder=5,
                )
                ax.annotate(
                    f"dim {d}\n{val:.1f}",
                    xy=(d, val),
                    xytext=(d + 80, val),
                    fontsize=7,
                    arrowprops=dict(
                        arrowstyle="->", color="black",
                        lw=0.8,
                    ),
                )

            ax.set_title(
                f"{label}" if row == 0
                else label,
                fontsize=9,
            )
            if row == 0:
                ax.set_title(
                    f"{col_labels[col]}\n{label}",
                    fontsize=10, fontweight="bold",
                )
            else:
                ax.set_title(label, fontsize=9)

            if col == 0:
                ax.set_ylabel(
                    f"Layer {li}\n|RMSNorm(h)|",
                    fontsize=10,
                )
            if row == len(sample_layers) - 1:
                ax.set_xlabel(
                    f"Hidden dimension (0..{hidden_dim-1})",
                )

    fig.suptitle(
        "|RMSNorm(hidden state)| across all dimensions\n"
        f"Sink dims = {sink_dims} (black dashed lines)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "dim_spikes.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")
    return path


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
        default="Describe what happens in this video.",
    )
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument(
        "--output-dir", default="sink_spikes_output",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sink_dims = [458, 2570]

    from mlx_vlm import load, generate
    from mlx_vlm.utils import load_config
    from mlx_vlm.video_generate import process_vision_info

    print(f"Loading {args.model}...")
    model, processor = load(args.model)
    config = load_config(args.model)
    patch_model(model)

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

    kwargs = {}
    for key in ["video_grid_thw", "image_grid_thw"]:
        if inputs.get(key) is not None:
            kwargs[key] = mx.array(inputs[key])

    model_config = model.config
    img_start, img_end = find_image_token_range(
        input_ids, model_config,
    )
    print(f"Vis tokens: [{img_start},{img_end})")

    CaptureStore.enable()
    kwargs["video"] = [args.video]
    kwargs["input_ids"] = input_ids
    kwargs["pixel_values"] = pixel_values
    kwargs["mask"] = mask_input
    kwargs["temperature"] = 0.0
    kwargs["max_tokens"] = args.max_tokens

    output = generate(
        model, processor, prompt=text,
        verbose=False, **kwargs,
    )
    print(f"Output: {output.text[:60]}...")
    CaptureStore.disable()

    hs_np = [np.array(h) for h in CaptureStore.hidden_states]
    print(f"Captured {len(hs_np)} layers, "
          f"dim={hs_np[0].shape[-1]}")

    path = plot_dim_spikes(
        hs_np, sink_dims, img_start, img_end,
        args.output_dir,
    )
    os.system(f"open '{path}'")


if __name__ == "__main__":
    main()
