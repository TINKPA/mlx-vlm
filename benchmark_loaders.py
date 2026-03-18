"""
Benchmark loaders for multi-benchmark ablation evaluation.

Provides unified loading for 3 image benchmarks (MMStar, POPE,
ScienceQA-IMG) and 3 MVBench video slices (temporal, spatial,
semantic).

Each loader returns List[dict] with keys:
  benchmark, question, candidates, gt, image, video_path, metadata
"""

import os
import re
from typing import List, Optional, Tuple

import numpy as np


# ── MVBench task groupings ──────────────────────────────

MVBENCH_TEMPORAL = [
    "action_sequence", "action_prediction",
    "action_localization", "state_change",
    "scene_transition",
]

MVBENCH_SPATIAL = [
    "object_existence", "object_interaction",
    "object_shuffle", "moving_direction",
    "moving_count", "moving_attribute",
]

MVBENCH_SEMANTIC = [
    "action_antonym", "counterfactual_inference",
    "unexpected_action", "fine_grained_action",
    "episodic_reasoning",
]


# ── Image benchmark loaders ─────────────────────────────

def load_mmstar(
    max_samples: int = 100, seed: int = 42,
) -> List[dict]:
    """Load MMStar val split from HuggingFace.

    Uses lazy image loading to avoid OOM — images are
    loaded on demand via _load_image callback, not kept
    in memory as PIL objects.
    """
    from datasets import load_dataset

    ds = load_dataset("Lin-Chen/MMStar", split="val")
    rng = np.random.RandomState(seed)

    indices = list(range(len(ds)))
    if max_samples < len(ds):
        indices = rng.choice(
            len(ds), max_samples, replace=False,
        ).tolist()

    items = []
    for i in indices:
        ex = ds[i]
        question = ex["question"]
        answer = ex["answer"]  # e.g. "A"
        candidates = _extract_mmstar_candidates(question)

        # Store index for lazy loading, not the image
        idx = i
        items.append({
            "benchmark": "mmstar",
            "question": question,
            "candidates": candidates,
            "gt": answer,
            "image": None,  # loaded lazily
            "video_path": None,
            "_load_image": lambda _idx=idx: ds[_idx][
                "image"
            ].convert("RGB"),
            "metadata": {
                "category": ex.get("category", ""),
                "l2_category": ex.get(
                    "l2_category", "",
                ),
            },
        })
    return items


def _extract_mmstar_candidates(question: str) -> List[str]:
    """Extract A/B/C/D choices from MMStar question text."""
    # Pattern: "A. text\nB. text\nC. text\nD. text"
    pattern = r'([A-E])\.\s*(.+?)(?=\n[A-E]\.|$)'
    matches = re.findall(pattern, question, re.DOTALL)
    if matches:
        return [m[0] for m in matches]  # just letters
    return ["A", "B", "C", "D"]


def load_pope(
    max_samples: int = 100, seed: int = 42,
) -> List[dict]:
    """Load POPE benchmark from HuggingFace.

    Lazy image loading to avoid OOM.
    """
    from datasets import load_dataset

    ds = load_dataset("lmms-lab/POPE", split="test")
    rng = np.random.RandomState(seed)

    indices = list(range(len(ds)))
    if max_samples < len(ds):
        indices = rng.choice(
            len(ds), max_samples, replace=False,
        ).tolist()

    items = []
    for i in indices:
        ex = ds[i]
        question = ex["question"]
        answer = ex["answer"]  # "yes" or "no"
        idx = i

        items.append({
            "benchmark": "pope",
            "question": question,
            "candidates": ["yes", "no"],
            "gt": answer.lower(),
            "image": None,
            "video_path": None,
            "_load_image": lambda _idx=idx: ds[_idx][
                "image"
            ].convert("RGB"),
            "metadata": {
                "source": ex.get("source", ""),
                "category": ex.get("category", ""),
            },
        })
    return items


def load_scienceqa_img(
    max_samples: int = 100, seed: int = 42,
) -> List[dict]:
    """Load ScienceQA test split, image-only questions.

    Lazy image loading to avoid OOM.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "derek-thomas/ScienceQA", split="test",
    )
    rng = np.random.RandomState(seed)

    # Filter to questions with images — check column,
    # not pixel data, to avoid loading images
    img_indices = [
        i for i in range(len(ds))
        if ds[i]["image"] is not None
    ]
    if max_samples < len(img_indices):
        img_indices = rng.choice(
            img_indices, max_samples, replace=False,
        ).tolist()

    letters = ["A", "B", "C", "D", "E"]
    items = []
    for i in img_indices:
        ex = ds[i]
        question = ex["question"]
        choices = ex["choices"]  # list of strings
        answer_idx = ex["answer"]  # int index

        # Format as MCQ
        choice_lines = []
        for ci, c in enumerate(choices):
            if ci < len(letters):
                choice_lines.append(
                    f"{letters[ci]}. {c}"
                )
        prompt = (
            f"{question}\n"
            + "\n".join(choice_lines) + "\n"
            + "Answer with the letter only "
            + f"({', '.join(letters[:len(choices)])})."
        )

        gt_letter = letters[answer_idx]

        idx = i
        items.append({
            "benchmark": "scienceqa",
            "question": prompt,
            "candidates": letters[:len(choices)],
            "gt": gt_letter,
            "image": None,
            "video_path": None,
            "_load_image": lambda _idx=idx: ds[_idx][
                "image"
            ].convert("RGB"),
            "metadata": {
                "subject": ex.get("subject", ""),
                "topic": ex.get("topic", ""),
            },
        })
    return items


# ── MVBench slice loader ─────────────────────────────────

def load_mvbench_slice(
    data_dir: str,
    task_list: List[str],
    max_per_task: int = 13,
    seed: int = 42,
) -> List[dict]:
    """Load a subset of MVBench tasks as a benchmark slice."""
    from sink_eviction_v10_mvbench import (
        load_mvbench_tasks, find_video, format_mcq_prompt,
    )

    all_tasks = load_mvbench_tasks(data_dir)
    rng = np.random.RandomState(seed)
    items = []

    for tname in task_list:
        if tname not in all_tasks:
            print(f"  WARN: task '{tname}' not found, skipping")
            continue
        qs = all_tasks[tname]
        n = min(max_per_task, len(qs))
        idx = rng.choice(len(qs), n, replace=False)

        for i in idx:
            item = qs[i]
            video_name = item["video"]
            video_path = find_video(video_name, data_dir)
            if video_path is None:
                continue

            prompt = format_mcq_prompt(
                item["question"], item["candidates"],
            )

            items.append({
                "benchmark": f"mvbench",
                "question": prompt,
                "candidates": item["candidates"],
                "gt": item["answer"],
                "image": None,
                "video_path": video_path,
                "metadata": {"task": tname},
            })

    return items


# ── Answer extraction (unified) ──────────────────────────

def extract_answer(
    benchmark: str, text: str, candidates: List[str],
) -> str:
    """Extract predicted answer from model output."""
    text = text.strip()

    if benchmark == "pope":
        # Yes/No — check first word
        first = text.split()[0].lower() if text else ""
        first = re.sub(r'[^a-z]', '', first)
        if first.startswith("yes"):
            return "yes"
        elif first.startswith("no"):
            return "no"
        # Fallback: search anywhere
        tl = text.lower()
        if "yes" in tl and "no" not in tl:
            return "yes"
        if "no" in tl and "yes" not in tl:
            return "no"
        return text[:20]

    # MCQ benchmarks: mmstar, scienceqa, mvbench
    letters = ["A", "B", "C", "D", "E"]

    # Direct letter match
    m = re.search(r'\b([A-E])\b', text)
    if m:
        letter = m.group(1)
        idx = letters.index(letter)
        if idx < len(candidates):
            return candidates[idx]

    # Try matching candidate text
    text_lower = text.lower()
    for i, c in enumerate(candidates):
        if c.lower() in text_lower:
            return c

    # Fallback: first character
    if text and text[0].upper() in letters:
        idx = letters.index(text[0].upper())
        if idx < len(candidates):
            return candidates[idx]

    return text[:50]


def check_correct(
    benchmark: str, pred: str, gt: str,
) -> bool:
    """Check if prediction matches ground truth."""
    if benchmark == "pope":
        return pred.lower().strip() == gt.lower().strip()

    if benchmark == "mmstar":
        # gt is a letter, pred may be letter or candidate
        return pred.upper() == gt.upper()

    # scienceqa: gt is letter, pred is candidate letter
    # mvbench: gt is candidate text, pred is candidate text
    return pred == gt
