#!/usr/bin/env python3
"""Summarize Front3D BS distribution for MPRI single-BS planning."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_DATASET_ROOT = "../data/Front3d_2988_CFR_Only"
DEFAULT_SPLIT_ID = "m117_single_scene_ue_level"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--dataset-root-env", default="BASELINE_FRONT3D_ROOT")
    parser.add_argument("--split-id", default=DEFAULT_SPLIT_ID)
    parser.add_argument("--output-root", type=Path, default=Path("runs/data_check"))
    parser.add_argument("--max-scenes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_root, args.dataset_root_env)
    split_root = dataset_root / "splits" / args.split_id
    scene_rows = load_scene_rows(split_root)
    if args.max_scenes is not None:
        scene_rows = scene_rows[: int(args.max_scenes)]
    if not scene_rows:
        raise ValueError(f"No scene rows found in {split_root}")

    per_scene = [summarize_scene(dataset_root, split_root, row) for row in scene_rows]
    summary = build_summary(dataset_root, split_root, per_scene)
    run_dir = make_run_dir(args.output_root, args.split_id)
    write_outputs(run_dir, summary, per_scene)
    print(run_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def resolve_dataset_root(dataset_root: Path | None, env_name: str) -> Path:
    root = os.environ.get(env_name, "") if env_name else ""
    if not root and dataset_root is not None:
        root = str(dataset_root)
    if not root:
        root = DEFAULT_DATASET_ROOT
    return Path(root).expanduser().resolve()


def load_scene_rows(split_root: Path) -> list[dict[str, str]]:
    path = split_root / "scene_splits.csv"
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_scene(dataset_root: Path, split_root: Path, scene_row: dict[str, str]) -> dict[str, Any]:
    scene_id = scene_row["scene_id"]
    scene_dir = split_root / scene_row["scene_dir"]
    h5_path = dataset_root / "scenes" / f"{scene_id}.h5"
    with h5py.File(h5_path, "r") as handle:
        scene_bs = np.unique(np.asarray(handle["bs_id"], dtype=np.int64))
        csi_power = np.asarray(handle["csi_power_db"], dtype=np.float32) if "csi_power_db" in handle else None
        labels_xy = np.asarray(handle["target_xy"], dtype=np.float32)
        bs_ids_all = np.asarray(handle["bs_id"], dtype=np.int64)
        bs_pos_all = np.asarray(handle["bs_pos"], dtype=np.float32) if "bs_pos" in handle else None

    split_stats: dict[str, Any] = {}
    split_bs_sets: dict[str, set[int]] = {}
    for split_name in ("train", "val", "test"):
        plan = load_link_plan(scene_dir / f"{split_name}_links.pt")
        bs = plan["bs_id"].astype(np.int64)
        ue = plan["ue_id"].astype(np.int64)
        split_bs_sets[split_name] = set(int(v) for v in np.unique(bs).tolist())
        per_bs_link_counts = Counter(int(v) for v in bs.tolist())
        split_stats[split_name] = {
            "link_count": int(bs.size),
            "ue_count": int(np.unique(ue).size),
            "bs_count": int(len(per_bs_link_counts)),
            "per_bs_link_min": int(min(per_bs_link_counts.values())) if per_bs_link_counts else 0,
            "per_bs_link_median": float(np.median(list(per_bs_link_counts.values()))) if per_bs_link_counts else 0.0,
            "per_bs_link_max": int(max(per_bs_link_counts.values())) if per_bs_link_counts else 0,
            "bs_ids": sorted(per_bs_link_counts),
        }
        if csi_power is not None and "local_link_index" in plan:
            idx = plan["local_link_index"].astype(np.int64)
            split_stats[split_name]["csi_power_db_median"] = float(np.median(csi_power[idx]))

    train_test_bs = split_bs_sets["train"] & split_bs_sets["test"]
    all_split_bs = split_bs_sets["train"] | split_bs_sets["val"] | split_bs_sets["test"]
    return {
        "scene_id": scene_id,
        "scene_dir": scene_row["scene_dir"],
        "scene_bs_count": int(scene_bs.size),
        "scene_bs_ids": [int(v) for v in scene_bs.tolist()],
        "m117_selected_link_count": int(scene_row["m117_selected_link_count"]),
        "m117_observed_ue_count": int(scene_row["m117_observed_ue_count"]),
        "train_link_count": split_stats["train"]["link_count"],
        "val_link_count": split_stats["val"]["link_count"],
        "test_link_count": split_stats["test"]["link_count"],
        "train_ue_count": split_stats["train"]["ue_count"],
        "val_ue_count": split_stats["val"]["ue_count"],
        "test_ue_count": split_stats["test"]["ue_count"],
        "train_bs_count": split_stats["train"]["bs_count"],
        "val_bs_count": split_stats["val"]["bs_count"],
        "test_bs_count": split_stats["test"]["bs_count"],
        "train_test_bs_count": int(len(train_test_bs)),
        "all_split_bs_count": int(len(all_split_bs)),
        "missing_test_bs_in_train": sorted(split_bs_sets["test"] - split_bs_sets["train"]),
        "train_per_bs_link_min": split_stats["train"]["per_bs_link_min"],
        "train_per_bs_link_median": split_stats["train"]["per_bs_link_median"],
        "train_per_bs_link_max": split_stats["train"]["per_bs_link_max"],
        "test_per_bs_link_min": split_stats["test"]["per_bs_link_min"],
        "test_per_bs_link_median": split_stats["test"]["per_bs_link_median"],
        "test_per_bs_link_max": split_stats["test"]["per_bs_link_max"],
        **select_representative_bs(train_plan=load_link_plan(scene_dir / "train_links.pt"), labels_xy=labels_xy, bs_ids_all=bs_ids_all, bs_pos_all=bs_pos_all, csi_power=csi_power),
    }


def select_representative_bs(
    train_plan: dict[str, np.ndarray],
    labels_xy: np.ndarray,
    bs_ids_all: np.ndarray,
    bs_pos_all: np.ndarray | None,
    csi_power: np.ndarray | None,
) -> dict[str, Any]:
    train_idx = train_plan["local_link_index"].astype(np.int64)
    train_bs = train_plan["bs_id"].astype(np.int64)
    train_center = np.median(labels_xy[train_idx, :2], axis=0)
    unique_bs = sorted(int(v) for v in np.unique(train_bs).tolist())
    median_rss_by_bs: dict[int, float] = {}
    central_distance_by_bs: dict[int, float] = {}
    for bs in unique_bs:
        mask = train_bs == bs
        idx = train_idx[mask]
        if csi_power is not None:
            median_rss_by_bs[bs] = float(np.median(csi_power[idx]))
        if bs_pos_all is not None:
            bs_pos = bs_pos_all[np.where(bs_ids_all == bs)[0][0], :2]
            central_distance_by_bs[bs] = float(np.linalg.norm(bs_pos - train_center))
    central_bs_id = min(central_distance_by_bs, key=central_distance_by_bs.get) if central_distance_by_bs else unique_bs[0]
    strongest_bs_id = max(median_rss_by_bs, key=median_rss_by_bs.get) if median_rss_by_bs else unique_bs[0]
    weakest_bs_id = min(median_rss_by_bs, key=median_rss_by_bs.get) if median_rss_by_bs else unique_bs[-1]
    return {
        "central_train_bs_id": int(central_bs_id),
        "central_train_bs_distance_m": central_distance_by_bs.get(int(central_bs_id), float("nan")),
        "strongest_train_median_rss_bs_id": int(strongest_bs_id),
        "strongest_train_median_rss_db": median_rss_by_bs.get(int(strongest_bs_id), float("nan")),
        "weakest_train_median_rss_bs_id": int(weakest_bs_id),
        "weakest_train_median_rss_db": median_rss_by_bs.get(int(weakest_bs_id), float("nan")),
    }


def load_link_plan(path: Path) -> dict[str, np.ndarray]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("Reading Front3D split link plans requires torch.") from exc
    plan = torch.load(path, map_location="cpu", weights_only=False)
    return {
        key: value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)
        for key, value in plan.items()
        if key != "metadata"
    }


def build_summary(dataset_root: Path, split_root: Path, per_scene: list[dict[str, Any]]) -> dict[str, Any]:
    scene_bs_counts = np.asarray([row["scene_bs_count"] for row in per_scene], dtype=np.int64)
    train_bs_counts = np.asarray([row["train_bs_count"] for row in per_scene], dtype=np.int64)
    test_bs_counts = np.asarray([row["test_bs_count"] for row in per_scene], dtype=np.int64)
    scene_bs_hist = Counter(int(v) for v in scene_bs_counts.tolist())
    missing = [row for row in per_scene if row["missing_test_bs_in_train"]]
    train_jobs_all_scene_bs = int(np.sum(scene_bs_counts))
    train_jobs_train_test_bs = int(np.sum([row["train_test_bs_count"] for row in per_scene]))
    train_jobs_one_bs_per_scene = int(len(per_scene))
    test_links = int(np.sum([row["test_link_count"] for row in per_scene]))
    return {
        "dataset_root": str(dataset_root),
        "split_root": str(split_root),
        "split_id": split_root.name,
        "scene_count": int(len(per_scene)),
        "total_scene_bs_pairs": train_jobs_all_scene_bs,
        "train_test_scene_bs_pairs": train_jobs_train_test_bs,
        "one_bs_per_scene_jobs": train_jobs_one_bs_per_scene,
        "all_bs_per_scene_jobs": train_jobs_all_scene_bs,
        "total_test_links": test_links,
        "scene_bs_count": describe(scene_bs_counts),
        "train_bs_count": describe(train_bs_counts),
        "test_bs_count": describe(test_bs_counts),
        "scene_bs_count_histogram": {str(k): int(v) for k, v in sorted(scene_bs_hist.items())},
        "scenes_with_test_bs_missing_in_train": int(len(missing)),
        "missing_test_bs_examples": [
            {
                "scene_id": row["scene_id"],
                "missing_test_bs_in_train": row["missing_test_bs_in_train"],
            }
            for row in missing[:10]
        ],
        "strategy_cost": {
            "all_scene_bs_single_bs_training_jobs": train_jobs_all_scene_bs,
            "train_test_scene_bs_single_bs_training_jobs": train_jobs_train_test_bs,
            "one_deterministic_bs_per_scene_training_jobs": train_jobs_one_bs_per_scene,
            "suggested_pilot": "one deterministic BS per scene, chosen by centrality or median train RSS; expand only if needed",
        },
    }


def describe(values: np.ndarray) -> dict[str, float | int]:
    return {
        "min": int(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p75": float(np.percentile(values, 75)),
        "max": int(np.max(values)),
    }


def make_run_dir(output_root: Path, split_id: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    base = output_root / f"front3d_bs_distribution_{split_id}-{stamp}"
    run_dir = base
    suffix = 1
    while run_dir.exists():
        run_dir = Path(f"{base}-{suffix:02d}")
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def write_outputs(run_dir: Path, summary: dict[str, Any], per_scene: list[dict[str, Any]]) -> None:
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fieldnames = [
        "scene_id",
        "scene_bs_count",
        "m117_selected_link_count",
        "m117_observed_ue_count",
        "train_link_count",
        "val_link_count",
        "test_link_count",
        "train_ue_count",
        "val_ue_count",
        "test_ue_count",
        "train_bs_count",
        "val_bs_count",
        "test_bs_count",
        "train_test_bs_count",
        "all_split_bs_count",
        "missing_test_bs_in_train",
        "train_per_bs_link_min",
        "train_per_bs_link_median",
        "train_per_bs_link_max",
        "test_per_bs_link_min",
        "test_per_bs_link_median",
        "test_per_bs_link_max",
        "central_train_bs_id",
        "central_train_bs_distance_m",
        "strongest_train_median_rss_bs_id",
        "strongest_train_median_rss_db",
        "weakest_train_median_rss_bs_id",
        "weakest_train_median_rss_db",
    ]
    with (run_dir / "scene_bs_distribution.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_scene:
            out = {key: row.get(key, "") for key in fieldnames}
            out["missing_test_bs_in_train"] = json.dumps(row["missing_test_bs_in_train"])
            writer.writerow(out)


if __name__ == "__main__":
    main()
