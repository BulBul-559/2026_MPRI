#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


DEFAULT_SOURCE_SPLIT = "m117_seed2026_scene_level"
DEFAULT_SINGLE_SCENE_SPLIT = "m117_single_scene_ue_level"
DEFAULT_SNR_DB = (50.0, 20.0, 10.0)
LINK_DATASETS_WITH_DERIVED_VALUES = {
    "cfr",
    "cfr_norm",
    "cfr_power_linear",
    "csi_power_db",
    "is_finite",
    "nan_count",
    "link_id",
    "source_local_index",
}


def default_source_root() -> Path:
    return Path(os.environ.get("BASELINE_FRONT3D_ROOT", "../data/Front3d_2988_CFR_Only"))


def default_output_root() -> Path:
    explicit = os.environ.get("BASELINE_FRONT3D_NOISE_ROOT")
    if explicit:
        return Path(explicit)
    return default_source_root() / "noise_sub_test_set"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Front3D test-scene noisy subset with MPRI SNR levels.")
    parser.add_argument("--source-root", type=Path, default=default_source_root())
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--source-split-id", default=DEFAULT_SOURCE_SPLIT)
    parser.add_argument("--single-scene-split-id", default=DEFAULT_SINGLE_SCENE_SPLIT)
    parser.add_argument("--snr-db", type=float, nargs="+", default=list(DEFAULT_SNR_DB))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--scene-limit", type=int, default=None)
    parser.add_argument("--chunk-links", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    source_split_dir = source_root / "splits" / args.source_split_id
    single_scene_root = source_root / "splits" / args.single_scene_split_id
    scene_ids = read_lines(source_split_dir / "test_scenes.txt")
    if args.scene_limit is not None:
        scene_ids = scene_ids[: int(args.scene_limit)]
    if not scene_ids:
        raise ValueError("No test scenes selected.")

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "scenes").mkdir(parents=True, exist_ok=True)
    (output_root / "splits").mkdir(parents=True, exist_ok=True)
    ensure_static_links(source_root, output_root)

    scene_summaries: list[dict[str, Any]] = []
    for scene_pos, scene_id in enumerate(scene_ids):
        src_h5 = source_root / "scenes" / f"{scene_id}.h5"
        dst_h5 = output_root / "scenes" / f"{scene_id}.h5"
        if args.overwrite and dst_h5.exists():
            dst_h5.unlink()
        if args.skip_existing and is_completed_scene(dst_h5, len(args.snr_db)):
            summary = summarize_noisy_scene(dst_h5, scene_id, scene_pos, args.snr_db)
            print(f"[skip] {scene_pos + 1}/{len(scene_ids)} {scene_id} already complete", flush=True)
        else:
            print(f"[build] {scene_pos + 1}/{len(scene_ids)} {scene_id}", flush=True)
            tmp_h5 = dst_h5.with_suffix(".h5.tmp")
            if tmp_h5.exists():
                tmp_h5.unlink()
            summary = build_noisy_scene(
                src_h5,
                tmp_h5,
                scene_id=scene_id,
                scene_pos=scene_pos,
                snr_db=args.snr_db,
                seed=args.seed,
                chunk_links=args.chunk_links,
            )
            os.replace(tmp_h5, dst_h5)
        scene_summaries.append(summary)

    write_dataset_manifest(output_root, source_root, args, scene_summaries)
    write_scene_level_split(output_root, source_root, args.source_split_id, scene_ids, args.snr_db, scene_summaries)
    write_single_scene_splits(output_root, source_root, args.single_scene_split_id, scene_ids, args.snr_db)
    write_link_index(output_root, scene_ids, args.snr_db)
    write_readme(output_root, source_root, args, scene_summaries)
    write_build_status(output_root, source_root, args, scene_summaries)
    print(json.dumps(build_final_report(output_root, source_root, args, scene_summaries), indent=2), flush=True)


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ensure_static_links(source_root: Path, output_root: Path) -> None:
    for name in ("assets", "obj"):
        target = output_root / name
        source = source_root / name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(source, target_is_directory=True)
    for name in ("environment.yml", "environment.lock.yml"):
        src = source_root / name
        if src.exists() and not (output_root / name).exists():
            shutil.copy2(src, output_root / name)


def is_completed_scene(path: Path, snr_count: int) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            return (
                "cfr" in f
                and "noise/snr_db" in f
                and int(f["noise/snr_db"].shape[0]) == int(f["cfr"].shape[0])
                and int(f.attrs.get("noise_snr_count", -1)) == snr_count
                and bool(f.attrs.get("noise_build_complete", False))
            )
    except OSError:
        return False


def build_noisy_scene(
    src_h5: Path,
    dst_h5: Path,
    *,
    scene_id: str,
    scene_pos: int,
    snr_db: list[float],
    seed: int,
    chunk_links: int,
) -> dict[str, Any]:
    with h5py.File(src_h5, "r") as src, h5py.File(dst_h5, "w") as dst:
        src_n = int(src["cfr"].shape[0])
        levels = np.asarray(snr_db, dtype=np.float32)
        out_n = src_n * int(levels.size)
        copy_attrs(src, dst)
        dst.attrs["noise_build_complete"] = False
        dst.attrs["noise_source_scene_path"] = str(src_h5)
        dst.attrs["noise_snr_count"] = int(levels.size)
        dst.attrs["noise_seed"] = int(seed)
        dst.attrs["noise_created_at"] = now_iso()
        copy_tree(src, dst, src_n=src_n, out_n=out_n, snr_count=int(levels.size))
        add_noise_group(dst, scene_id, src_n, levels, seed)
        update_scalar_metadata(dst, out_n, levels)
        write_noisy_cfr_and_derived(src, dst, scene_id, levels, seed, chunk_links)
        dst.attrs["noise_build_complete"] = True
    return summarize_noisy_scene(dst_h5, scene_id, scene_pos, snr_db)


def copy_tree(src_group: h5py.Group, dst_group: h5py.Group, *, src_n: int, out_n: int, snr_count: int) -> None:
    for name, obj in src_group.items():
        if isinstance(obj, h5py.Group):
            child = dst_group.create_group(name)
            copy_attrs(obj, child)
            copy_tree(obj, child, src_n=src_n, out_n=out_n, snr_count=snr_count)
            continue
        if full_name(obj) in LINK_DATASETS_WITH_DERIVED_VALUES:
            continue
        if is_link_dataset(obj, src_n):
            data = obj[()]
            expanded = np.repeat(data, snr_count, axis=0)
            create_dataset_like(dst_group, name, obj, data=expanded)
        else:
            create_dataset_like(dst_group, name, obj, data=obj[()])


def full_name(obj: h5py.Dataset) -> str:
    return obj.name.lstrip("/")


def is_link_dataset(ds: h5py.Dataset, src_n: int) -> bool:
    return len(ds.shape) >= 1 and int(ds.shape[0]) == int(src_n)


def copy_attrs(src: h5py.Dataset | h5py.Group | h5py.File, dst: h5py.Dataset | h5py.Group | h5py.File) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def create_dataset_like(group: h5py.Group, name: str, src: h5py.Dataset, *, data: np.ndarray | bytes | str | int | float) -> h5py.Dataset:
    kwargs: dict[str, Any] = {}
    if src.shape != () and src.compression is not None:
        kwargs["compression"] = src.compression
        if src.compression_opts is not None:
            kwargs["compression_opts"] = src.compression_opts
        if src.shuffle:
            kwargs["shuffle"] = src.shuffle
        if src.fletcher32:
            kwargs["fletcher32"] = src.fletcher32
        if src.chunks is not None and tuple(src.chunks) == tuple(np.shape(data)):
            kwargs["chunks"] = src.chunks
    try:
        ds = group.create_dataset(name, data=data, dtype=src.dtype, **kwargs)
    except TypeError:
        ds = group.create_dataset(name, data=data, **kwargs)
    copy_attrs(src, ds)
    return ds


def add_noise_group(dst: h5py.File, scene_id: str, src_n: int, snr_db: np.ndarray, seed: int) -> None:
    noise = dst.create_group("noise")
    noise.attrs["contract_name"] = "front3d_cfr_offline_awgn_noise"
    noise.attrs["schema_version"] = 1
    noise.attrs["source"] = "MPRI SNR augmentation levels applied offline to complex CFR"
    snr = np.tile(snr_db, src_n).astype(np.float32)
    level_index = np.tile(np.arange(snr_db.size, dtype=np.int16), src_n)
    source_index = np.repeat(np.arange(src_n, dtype=np.int64), snr_db.size)
    noise.create_dataset("snr_db", data=snr, compression="gzip")
    noise.create_dataset("noise_level_index", data=level_index, compression="gzip")
    noise.create_dataset("source_local_link_index", data=source_index, compression="gzip")
    noise.create_dataset("source_link_id", data=source_index, compression="gzip")
    noise.create_dataset("scene_id", data=np.asarray([scene_id], dtype=h5py.string_dtype("utf-8")))
    noise.create_dataset("seed", data=np.asarray([seed], dtype=np.int64))


def update_scalar_metadata(dst: h5py.File, out_n: int, snr_db: np.ndarray) -> None:
    if "scene/num_links" in dst:
        dst["scene/num_links"][()] = np.int64(out_n)
    if "scene_id" in dst:
        dst["scene_id"].attrs["noise_sub_test_set"] = True
    for path in ("meta/created_at",):
        if path in dst:
            replace_scalar_string(dst, path, now_iso())
    for path, value in {
        "meta/noise_contract_name": "front3d_cfr_offline_awgn_noise",
        "meta/noise_snr_db": json.dumps([float(v) for v in snr_db.tolist()]),
        "meta/noise_source": "MPRI SNR augmentation applied offline to complex CFR",
    }.items():
        write_scalar_string(dst, path, value)


def replace_scalar_string(dst: h5py.File, path: str, value: str) -> None:
    parent_name, dataset_name = path.rsplit("/", 1)
    parent = dst[parent_name]
    del parent[dataset_name]
    parent.create_dataset(dataset_name, data=value, dtype=h5py.string_dtype("utf-8"))


def write_scalar_string(dst: h5py.File, path: str, value: str) -> None:
    parent_name, dataset_name = path.rsplit("/", 1)
    parent = dst.require_group(parent_name)
    if dataset_name in parent:
        del parent[dataset_name]
    parent.create_dataset(dataset_name, data=value, dtype=h5py.string_dtype("utf-8"))


def write_noisy_cfr_and_derived(
    src: h5py.File,
    dst: h5py.File,
    scene_id: str,
    snr_db: np.ndarray,
    seed: int,
    chunk_links: int,
) -> None:
    src_cfr = src["cfr"]
    src_n = int(src_cfr.shape[0])
    snr_count = int(snr_db.size)
    out_shape = (src_n * snr_count, *src_cfr.shape[1:])
    dst_cfr = dst.create_dataset("cfr", shape=out_shape, dtype=src_cfr.dtype)
    copy_attrs(src_cfr, dst_cfr)
    dst.create_dataset("cfr_power_linear", shape=(out_shape[0],), dtype=np.float32, compression="gzip")
    dst.create_dataset("csi_power_db", shape=(out_shape[0],), dtype=np.float32, compression="gzip")
    dst.create_dataset("cfr_norm", shape=(out_shape[0],), dtype=np.float32, compression="gzip")
    dst.create_dataset("nan_count", shape=(out_shape[0],), dtype=np.int32)
    dst.create_dataset("is_finite", shape=(out_shape[0],), dtype=bool)
    dst.create_dataset("link_id", data=np.arange(out_shape[0], dtype=np.int64))
    dst.create_dataset("source_local_index", data=np.arange(out_shape[0], dtype=np.int32))
    for start in range(0, src_n, chunk_links):
        end = min(src_n, start + chunk_links)
        base = np.asarray(src_cfr[start:end], dtype=np.complex64)
        out = np.empty((base.shape[0] * snr_count, *base.shape[1:]), dtype=np.complex64)
        for level_i, level in enumerate(snr_db.tolist()):
            rng = np.random.default_rng(noise_seed(seed, scene_id, int(level_i), int(start)))
            out[level_i::snr_count] = add_awgn_batch(base, float(level), rng)
        dst_slice = slice(start * snr_count, end * snr_count)
        dst_cfr[dst_slice] = out
        power = np.nanmean(np.abs(out) ** 2, axis=(1, 2, 3)).astype(np.float32)
        finite = np.isfinite(out)
        dst["cfr_power_linear"][dst_slice] = power
        dst["csi_power_db"][dst_slice] = (10.0 * np.log10(power + 1e-30)).astype(np.float32)
        dst["cfr_norm"][dst_slice] = np.linalg.norm(out.reshape(out.shape[0], -1), axis=1).astype(np.float32)
        dst["nan_count"][dst_slice] = np.size(out[0]) - finite.reshape(out.shape[0], -1).sum(axis=1).astype(np.int32)
        dst["is_finite"][dst_slice] = finite.reshape(out.shape[0], -1).all(axis=1)


def noise_seed(seed: int, scene_id: str, level_i: int, chunk_start: int) -> int:
    digest = hashlib.sha256(f"{seed}:{scene_id}:{level_i}:{chunk_start}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def add_awgn_batch(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal_power = np.nanmean(np.abs(x) ** 2, axis=(1, 2, 3), keepdims=True).astype(np.float32)
    safe_power = np.where(np.isfinite(signal_power) & (signal_power > 0.0), signal_power, 0.0)
    noise_power = safe_power / (10.0 ** (float(snr_db) / 10.0))
    noise_scale = np.sqrt(noise_power / 2.0).astype(np.float32)
    noise = noise_scale * (
        rng.standard_normal(x.shape, dtype=np.float32) + 1j * rng.standard_normal(x.shape, dtype=np.float32)
    )
    return (x + noise.astype(np.complex64)).astype(np.complex64)


def summarize_noisy_scene(path: Path, scene_id: str, scene_pos: int, snr_db: list[float]) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        csi = np.asarray(f["csi_power_db"])
        valid = np.asarray(f["valid_mask"] if "valid_mask" in f else f["is_finite"], dtype=bool)
        return {
            "scene_id": scene_id,
            "scene_row": int(scene_pos),
            "h5_path": f"scenes/{scene_id}.h5",
            "source_num_links": int(f["noise/source_local_link_index"].shape[0] // len(snr_db)),
            "num_links": int(f["cfr"].shape[0]),
            "valid_link_count": int(valid.sum()),
            "cfr_shape": list(f["cfr"].shape),
            "dtype": str(f["cfr"].dtype),
            "output_bytes": int(path.stat().st_size),
            "power_min_db": float(np.nanmin(csi)) if csi.size else None,
            "power_max_db": float(np.nanmax(csi)) if csi.size else None,
            "snr_db": [float(v) for v in snr_db],
            "status": "completed",
        }


def write_dataset_manifest(
    output_root: Path,
    source_root: Path,
    args: argparse.Namespace,
    scene_summaries: list[dict[str, Any]],
) -> None:
    manifest = {
        "contract_name": "front3d_cfr_noise_sub_test_set_manifest",
        "schema_version": 1,
        "created_at": now_iso(),
        "source_dataset_root": str(source_root),
        "source_split_id": args.source_split_id,
        "source_split": "test",
        "single_scene_split_id": args.single_scene_split_id,
        "output_root": str(output_root),
        "noise": {
            "type": "complex_awgn",
            "snr_db": [float(v) for v in args.snr_db],
            "seed": int(args.seed),
            "insertion_point": "complex CFR",
            "source_policy": "MPRI offline SNR augmentation levels",
        },
        "scene_count": len(scene_summaries),
        "num_links": int(sum(item["num_links"] for item in scene_summaries)),
        "source_num_links": int(sum(item["source_num_links"] for item in scene_summaries)),
        "scenes": scene_summaries,
        "link_index_pt": "link_index.pt",
        "link_index_parquet": "link_index.parquet",
    }
    write_json(output_root / "manifest.json", manifest)


def write_scene_level_split(
    output_root: Path,
    source_root: Path,
    source_split_id: str,
    scene_ids: list[str],
    snr_db: list[float],
    scene_summaries: list[dict[str, Any]],
) -> None:
    src_dir = source_root / "splits" / source_split_id
    dst_dir = output_root / "splits" / source_split_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    write_lines(dst_dir / "all_scenes.txt", scene_ids)
    write_lines(dst_dir / "test_scenes.txt", scene_ids)
    write_lines(dst_dir / "train_scenes.txt", [])
    write_lines(dst_dir / "val_scenes.txt", [])
    write_scene_split_files(dst_dir, scene_ids, scene_summaries)
    global_offsets = compute_global_offsets(output_root, scene_ids)
    write_expanded_scene_link_plan(
        src_dir / "test_links.pt",
        output_root,
        dst_dir / "test_links.pt",
        snr_db,
        scene_ids,
        global_offsets,
        source_split_id=source_split_id,
    )
    write_empty_link_plan(dst_dir / "train_links.pt", split="train", snr_db=snr_db)
    write_empty_link_plan(dst_dir / "val_links.pt", split="val", snr_db=snr_db)
    source_test_count = count_scene_link_plan(src_dir / "test_links.pt", len(scene_ids))
    stats = {
        "kind": "noise_sub_test_set_scene_level_split_stats",
        "version": 1,
        "source_split_id": source_split_id,
        "source_split": "test",
        "scene_count": len(scene_ids),
        "snr_db": [float(v) for v in snr_db],
        "source_num_links": int(sum(item["source_num_links"] for item in scene_summaries)),
        "num_links": int(sum(item["num_links"] for item in scene_summaries)),
        "splits": {
            "train": {"scene_count": 0, "selected_link_count": 0},
            "val": {"scene_count": 0, "selected_link_count": 0},
            "test": {
                "scene_count": len(scene_ids),
                "source_selected_link_count": int(source_test_count),
                "selected_link_count": int(source_test_count * len(snr_db)),
            },
        },
    }
    write_json(dst_dir / "split_stats.json", stats)
    manifest = {
        "contract_name": "front3d_cfr_noise_sub_test_set_scene_level_split",
        "schema_version": 1,
        "source_split_id": source_split_id,
        "source_split": "test",
        "split_id": source_split_id,
        "files": {
            "all_scenes": "all_scenes.txt",
            "scene_split_csv": "scene_split.csv",
            "scene_split_jsonl": "scene_split.jsonl",
            "split_stats": "split_stats.json",
            "test_links": "test_links.pt",
            "test_scenes": "test_scenes.txt",
            "train_links": "train_links.pt",
            "train_scenes": "train_scenes.txt",
            "val_links": "val_links.pt",
            "val_scenes": "val_scenes.txt",
        },
    }
    write_json(dst_dir / "split_manifest.json", manifest)
    write_split_readme(dst_dir / "README.md", "Scene-level split over noisy m117 source test scenes.")
    write_checksums(dst_dir)


def write_scene_split_files(dst_dir: Path, scene_ids: list[str], scene_summaries: list[dict[str, Any]]) -> None:
    rows = []
    summary_by_scene = {item["scene_id"]: item for item in scene_summaries}
    for i, scene_id in enumerate(scene_ids):
        item = summary_by_scene[scene_id]
        rows.append(
            {
                "scene_id": scene_id,
                "split": "test",
                "scene_row": i,
                "h5_path": f"scenes/{scene_id}.h5",
                "num_links": item["num_links"],
                "source_num_links": item["source_num_links"],
                "valid_link_count": item["valid_link_count"],
                "output_bytes": item["output_bytes"],
            }
        )
    write_csv(dst_dir / "scene_split.csv", rows)
    write_jsonl(dst_dir / "scene_split.jsonl", rows)


def write_expanded_scene_link_plan(
    src_path: Path,
    output_root: Path,
    dst_path: Path,
    snr_db: list[float],
    scene_ids: list[str],
    global_offsets: dict[str, int],
    *,
    source_split_id: str,
) -> None:
    source = filter_scene_link_plan(torch.load(src_path, map_location="cpu"), max_scene_count=len(scene_ids))
    record_index = source["record_index"].to(torch.long)
    local_index = source["local_link_index"].to(torch.long)
    snr_count = len(snr_db)
    out_local = expand_indices(local_index, snr_count)
    out_record = repeat_tensor(record_index, snr_count)
    csi = csi_for_expanded_scene_plan(output_root, source_split_id, out_record.tolist(), out_local.tolist())
    out = expand_link_plan_common(source, snr_db, split="test")
    out["record_index"] = out_record
    out["local_link_index"] = out_local
    out["global_link_id"] = global_ids_from_scene_rows(out_record, out_local, scene_ids, global_offsets)
    out["source_global_link_id"] = repeat_tensor(source["global_link_id"].to(torch.long), snr_count)
    out["source_local_link_index"] = repeat_tensor(local_index, snr_count)
    out["snr_db"] = repeat_snr(len(local_index), snr_db)
    out["noise_level_index"] = repeat_noise_level(len(local_index), snr_count)
    out["csi_power_db"] = torch.as_tensor(csi, dtype=torch.float32)
    out["metadata"]["selected_link_count"] = int(out_local.numel())
    out["metadata"]["source_selected_link_count"] = int(local_index.numel())
    out["metadata"]["noise_snr_db"] = [float(v) for v in snr_db]
    torch.save(out, dst_path)


def filter_scene_link_plan(source: dict[str, Any], *, max_scene_count: int) -> dict[str, Any]:
    record_index = source["record_index"].to(torch.long)
    keep = record_index < int(max_scene_count)
    filtered: dict[str, Any] = {"metadata": dict(source.get("metadata", {}))}
    for key, value in source.items():
        if key == "metadata":
            continue
        if torch.is_tensor(value) and value.shape[:1] == record_index.shape[:1]:
            filtered[key] = value[keep]
        else:
            filtered[key] = value
    filtered["metadata"]["source_selected_link_count_before_scene_filter"] = int(record_index.numel())
    filtered["metadata"]["source_selected_link_count"] = int(filtered["local_link_index"].numel())
    return filtered


def csi_for_expanded_scene_plan(
    output_root: Path,
    source_split_id: str,
    scene_rows: list[int],
    local_indices: list[int],
) -> np.ndarray:
    if len(scene_rows) != len(local_indices):
        raise ValueError(
            "expanded scene row and local index arrays must have the same length: "
            f"{len(scene_rows)} != {len(local_indices)}"
        )
    scene_ids = read_lines(output_root / "splits" / source_split_id / "test_scenes.txt")
    result = np.empty(len(local_indices), dtype=np.float32)
    by_scene: dict[int, list[tuple[int, int]]] = {}
    for out_pos, (scene_row, local_idx) in enumerate(zip(scene_rows, local_indices)):
        by_scene.setdefault(int(scene_row), []).append((out_pos, int(local_idx)))
    for scene_row, pairs in by_scene.items():
        scene_id = scene_ids[scene_row]
        with h5py.File(output_root / "scenes" / f"{scene_id}.h5", "r") as f:
            values = np.asarray(f["csi_power_db"])
        for out_pos, local_idx in pairs:
            result[out_pos] = values[local_idx]
    return result


def write_empty_link_plan(path: Path, *, split: str, snr_db: list[float]) -> None:
    empty_long = torch.empty(0, dtype=torch.long)
    empty_float = torch.empty(0, dtype=torch.float32)
    obj = {
        "metadata": {
            "kind": "scene_dataset_link_index_plan",
            "version": 1,
            "split": split,
            "selected_link_count": 0,
            "source_selected_link_count": 0,
            "noise_snr_db": [float(v) for v in snr_db],
        },
        "record_index": empty_long,
        "local_link_index": empty_long,
        "ue_id": empty_long,
        "bs_id": empty_long,
        "global_link_id": empty_long,
        "source_global_link_id": empty_long,
        "source_local_link_index": empty_long,
        "csi_power_db": empty_float,
        "snr_db": empty_float,
        "noise_level_index": empty_long,
    }
    torch.save(obj, path)


def write_single_scene_splits(
    output_root: Path,
    source_root: Path,
    single_scene_split_id: str,
    scene_ids: list[str],
    snr_db: list[float],
) -> None:
    dst_root = output_root / "splits" / single_scene_split_id
    dst_root.mkdir(parents=True, exist_ok=True)
    source_split_root = source_root / "splits" / single_scene_split_id
    global_offsets = compute_global_offsets(output_root, scene_ids)
    for scene_id in scene_ids:
        src_dir = source_split_root / f"{scene_id}_seed2026_ue_level"
        dst_dir = dst_root / src_dir.name
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir, ignore=shutil.ignore_patterns("*.pt", "split_stats.json", "split_manifest.json", "ue_split.csv", "ue_split.jsonl", "checksums.sha256", "README.md"))
        for split in ("train", "val", "test"):
            write_expanded_single_scene_link_plan(
                src_dir / f"{split}_links.pt",
                output_root / "scenes" / f"{scene_id}.h5",
                dst_dir / f"{split}_links.pt",
                split,
                snr_db,
                global_offset=global_offsets[scene_id],
            )
        rewrite_ue_split(src_dir / "ue_split.csv", dst_dir / "ue_split.csv", snr_count=len(snr_db))
        rewrite_ue_split_jsonl(src_dir / "ue_split.jsonl", dst_dir / "ue_split.jsonl", snr_count=len(snr_db))
        rewrite_single_scene_stats(src_dir / "split_stats.json", dst_dir / "split_stats.json", snr_db)
        rewrite_single_scene_manifest(src_dir / "split_manifest.json", dst_dir / "split_manifest.json", snr_db)
        write_split_readme(dst_dir / "README.md", f"UE-level split for noisy scene {scene_id}; UE membership matches the source split.")
        write_checksums(dst_dir)
    write_json(
        dst_root / "split_manifest.json",
        {
            "contract_name": "front3d_cfr_noise_sub_test_set_single_scene_ue_level_collection",
            "schema_version": 1,
            "source_split_id": single_scene_split_id,
            "scene_count": len(scene_ids),
            "snr_db": [float(v) for v in snr_db],
        },
    )
    write_lines(dst_root / "all_scenes.txt", scene_ids)
    write_split_readme(dst_root / "README.md", "Collection of per-scene UE-level splits for noisy m117 source test scenes.")
    write_checksums(dst_root)


def write_expanded_single_scene_link_plan(
    src_path: Path,
    scene_h5: Path,
    dst_path: Path,
    split: str,
    snr_db: list[float],
    *,
    global_offset: int,
) -> None:
    source = torch.load(src_path, map_location="cpu")
    local_index = source["local_link_index"].to(torch.long)
    snr_count = len(snr_db)
    out = expand_link_plan_common(source, snr_db, split=split)
    out["local_link_index"] = expand_indices(local_index, snr_count)
    out["global_link_id"] = out["local_link_index"] + int(global_offset)
    out["source_global_link_id"] = repeat_tensor(source["global_link_id"].to(torch.long), snr_count)
    out["source_local_link_index"] = repeat_tensor(local_index, snr_count)
    out["snr_db"] = repeat_snr(len(local_index), snr_db)
    out["noise_level_index"] = repeat_noise_level(len(local_index), snr_count)
    with h5py.File(scene_h5, "r") as f:
        out["csi_power_db"] = torch.as_tensor(np.asarray(f["csi_power_db"])[out["local_link_index"].numpy()], dtype=torch.float32)
    out["metadata"]["selected_link_count"] = int(out["local_link_index"].numel())
    out["metadata"]["source_selected_link_count"] = int(local_index.numel())
    out["metadata"]["noise_snr_db"] = [float(v) for v in snr_db]
    torch.save(out, dst_path)


def expand_link_plan_common(source: dict[str, Any], snr_db: list[float], *, split: str) -> dict[str, Any]:
    out: dict[str, Any] = {"metadata": dict(source.get("metadata", {}))}
    out["metadata"]["split"] = split
    out["metadata"]["noise_contract_name"] = "front3d_cfr_offline_awgn_noise"
    for key, value in source.items():
        if key == "metadata":
            continue
        if key in {"local_link_index", "global_link_id", "csi_power_db"}:
            continue
        if torch.is_tensor(value):
            out[key] = repeat_tensor(value, len(snr_db))
        else:
            out[key] = value
    return out


def compute_global_offsets(output_root: Path, scene_ids: list[str]) -> dict[str, int]:
    offsets: dict[str, int] = {}
    offset = 0
    for scene_id in scene_ids:
        offsets[scene_id] = offset
        with h5py.File(output_root / "scenes" / f"{scene_id}.h5", "r") as f:
            offset += int(f["cfr"].shape[0])
    return offsets


def global_ids_from_scene_rows(
    record_index: torch.Tensor,
    local_link_index: torch.Tensor,
    scene_ids: list[str],
    global_offsets: dict[str, int],
) -> torch.Tensor:
    offsets = torch.empty_like(local_link_index, dtype=torch.long)
    for row in torch.unique(record_index).tolist():
        scene_id = scene_ids[int(row)]
        offsets[record_index == int(row)] = int(global_offsets[scene_id])
    return offsets + local_link_index.to(torch.long)


def repeat_tensor(x: torch.Tensor, repeats: int) -> torch.Tensor:
    return x.repeat_interleave(repeats, dim=0)


def expand_indices(x: torch.Tensor, repeats: int) -> torch.Tensor:
    offsets = torch.arange(repeats, dtype=x.dtype)
    return (x.repeat_interleave(repeats, dim=0) * repeats + offsets.repeat(x.numel())).to(x.dtype)


def repeat_snr(source_count: int, snr_db: list[float]) -> torch.Tensor:
    return torch.tensor(snr_db, dtype=torch.float32).repeat(source_count)


def repeat_noise_level(source_count: int, snr_count: int) -> torch.Tensor:
    return torch.arange(snr_count, dtype=torch.long).repeat(source_count)


def rewrite_ue_split(src: Path, dst: Path, *, snr_count: int) -> None:
    rows = []
    with src.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = dict(row)
            row["selected_link_count"] = str(int(row["selected_link_count"]) * snr_count)
            rows.append(row)
    write_csv(dst, rows)


def rewrite_ue_split_jsonl(src: Path, dst: Path, *, snr_count: int) -> None:
    rows = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["selected_link_count"] = int(row["selected_link_count"]) * snr_count
        rows.append(row)
    write_jsonl(dst, rows)


def rewrite_single_scene_stats(src: Path, dst: Path, snr_db: list[float]) -> None:
    stats = json.loads(src.read_text(encoding="utf-8"))
    stats["kind"] = "noise_sub_test_set_single_scene_ue_level_split_stats"
    stats["noise_snr_db"] = [float(v) for v in snr_db]
    stats["source_stats_path"] = str(src)
    for split_info in stats.get("splits", {}).values():
        source_count = int(split_info.get("selected_link_count", 0))
        split_info["source_selected_link_count"] = source_count
        split_info["selected_link_count"] = source_count * len(snr_db)
    scene_meta = stats.get("scene_metadata", {})
    if "num_links" in scene_meta:
        scene_meta["source_num_links"] = int(scene_meta["num_links"])
        scene_meta["num_links"] = int(scene_meta["num_links"]) * len(snr_db)
    write_json(dst, stats)


def rewrite_single_scene_manifest(src: Path, dst: Path, snr_db: list[float]) -> None:
    manifest = json.loads(src.read_text(encoding="utf-8"))
    manifest["contract_name"] = "front3d_cfr_noise_sub_test_set_single_scene_ue_level_split"
    manifest["noise_snr_db"] = [float(v) for v in snr_db]
    manifest["source_manifest_path"] = str(src)
    validation = manifest.setdefault("validation", {})
    validation["ue_sets_disjoint"] = True
    validation["noise_expansion_preserves_ue_membership"] = True
    write_json(dst, manifest)


def write_link_index(output_root: Path, scene_ids: list[str], snr_db: list[float]) -> None:
    parts: dict[str, list[torch.Tensor]] = {
        "global_link_id": [],
        "scene_row": [],
        "local_link_index": [],
        "source_local_link_index": [],
        "bs_id": [],
        "ue_id": [],
        "rx_pos": [],
        "tx_pos": [],
        "target_xy": [],
        "csi_power_db": [],
        "valid_mask": [],
        "ue_room_index": [],
        "bs_room_index": [],
        "target_room_index": [],
        "target_room_type_id": [],
        "target_room_area_m2": [],
        "snr_db": [],
        "noise_level_index": [],
    }
    global_offset = 0
    rows_for_csv: list[dict[str, Any]] = []
    for scene_row, scene_id in enumerate(scene_ids):
        with h5py.File(output_root / "scenes" / f"{scene_id}.h5", "r") as f:
            n = int(f["cfr"].shape[0])
            local = torch.arange(n, dtype=torch.long)
            parts["global_link_id"].append(torch.arange(global_offset, global_offset + n, dtype=torch.long))
            parts["scene_row"].append(torch.full((n,), scene_row, dtype=torch.int32))
            parts["local_link_index"].append(local)
            parts["source_local_link_index"].append(torch.as_tensor(np.asarray(f["noise/source_local_link_index"]), dtype=torch.long))
            for key in ("bs_id", "ue_id", "valid_mask"):
                parts[key].append(torch.as_tensor(np.asarray(f[key])))
            for key in ("rx_pos", "tx_pos", "target_xy", "csi_power_db"):
                parts[key].append(torch.as_tensor(np.asarray(f[key])))
            for key in ("ue_room_index", "bs_room_index", "target_room_index", "target_room_type_id", "target_room_area_m2"):
                h5_key = f"link/{key}" if f"link/{key}" in f else key
                parts[key].append(torch.as_tensor(np.asarray(f[h5_key])))
            parts["snr_db"].append(torch.as_tensor(np.asarray(f["noise/snr_db"]), dtype=torch.float32))
            parts["noise_level_index"].append(torch.as_tensor(np.asarray(f["noise/noise_level_index"]), dtype=torch.long))
            global_offset += n
            rows_for_csv.append(
                {
                    "scene_id": scene_id,
                    "scene_row": scene_row,
                    "num_links": n,
                    "source_num_links": int(f["noise/source_local_link_index"].shape[0] // len(snr_db)),
                    "h5_path": f"scenes/{scene_id}.h5",
                }
            )
    link_index: dict[str, Any] = {
        "schema_version": "0.2-noise-sub-test-set",
        "contract_name": "sionna_cfr_scene_dataset_link_index_noise_sub_test_set",
        "scene_ids": scene_ids,
        "scene_h5_paths": [f"scenes/{scene_id}.h5" for scene_id in scene_ids],
        "noise_snr_db": [float(v) for v in snr_db],
    }
    for key, values in parts.items():
        link_index[key] = torch.cat(values, dim=0) if values else torch.empty(0)
    torch.save(link_index, output_root / "link_index.pt")
    write_csv(output_root / "link_index_scene_summary.csv", rows_for_csv)
    maybe_write_link_index_parquet(output_root, link_index)


def maybe_write_link_index_parquet(output_root: Path, link_index: dict[str, Any]) -> None:
    try:
        import pandas as pd

        data = {
            "global_link_id": link_index["global_link_id"].numpy(),
            "scene_row": link_index["scene_row"].numpy(),
            "local_link_index": link_index["local_link_index"].numpy(),
            "source_local_link_index": link_index["source_local_link_index"].numpy(),
            "bs_id": link_index["bs_id"].numpy(),
            "ue_id": link_index["ue_id"].numpy(),
            "csi_power_db": link_index["csi_power_db"].numpy(),
            "valid_mask": link_index["valid_mask"].numpy(),
            "snr_db": link_index["snr_db"].numpy(),
            "noise_level_index": link_index["noise_level_index"].numpy(),
        }
        pd.DataFrame(data).to_parquet(output_root / "link_index.parquet", index=False)
    except Exception as exc:
        (output_root / "link_index.parquet.SKIPPED.txt").write_text(
            f"Parquet export skipped because pandas parquet engine is unavailable or failed: {exc}\n",
            encoding="utf-8",
        )


def count_scene_link_plan(path: Path, max_scene_count: int | None = None) -> int:
    source = torch.load(path, map_location="cpu")
    if max_scene_count is not None and "record_index" in source:
        return int((source["record_index"].to(torch.long) < int(max_scene_count)).sum().item())
    return int(source["local_link_index"].numel())


def write_readme(output_root: Path, source_root: Path, args: argparse.Namespace, scene_summaries: list[dict[str, Any]]) -> None:
    text = f"""# Front3D CFR Noise Sub Test Set

This dataset contains the `test` scenes from `{args.source_split_id}` in `{source_root}` with offline complex AWGN applied at SNR levels `{[float(v) for v in args.snr_db]}`.

Each source link is expanded into one link per SNR level. Per-scene UE split membership is copied from `{args.single_scene_split_id}` and link indices are remapped as `new_local_link_index = source_local_link_index * {len(args.snr_db)} + noise_level_index`.

Generated at: {now_iso()}

Scene count: {len(scene_summaries)}
Source links: {sum(item['source_num_links'] for item in scene_summaries)}
Noisy links: {sum(item['num_links'] for item in scene_summaries)}

Static `assets/` and `obj/` directories are symlinks to the source dataset.
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def write_build_status(
    output_root: Path,
    source_root: Path,
    args: argparse.Namespace,
    scene_summaries: list[dict[str, Any]],
) -> None:
    rows = []
    for scene_order, item in enumerate(scene_summaries, start=1):
        h5_path = output_root / item["h5_path"]
        tmp_path = h5_path.with_suffix(h5_path.suffix + ".tmp")
        rows.append(
            {
                "scene_order": scene_order,
                "scene_id": item["scene_id"],
                "status": item.get("status", "completed"),
                "h5_path": str(h5_path),
                "tmp_path": str(tmp_path) if tmp_path.exists() else "",
                "size_bytes": int(item["output_bytes"]),
                "source_num_links": int(item["source_num_links"]),
                "num_links": int(item["num_links"]),
                "noise_snr_count": len(args.snr_db),
                "error": "",
            }
        )
    write_csv(output_root / "build_status.csv", rows)

    report = build_final_report(output_root, source_root, args, scene_summaries)
    tmp_count = sum(1 for path in (output_root / "scenes").glob("*.tmp"))
    state = "complete" if tmp_count == 0 else "incomplete"
    text = f"""# Noise Sub Test Set Build Status

Last updated: {now_iso()}

This offline noisy subset is currently **{state}** for the `{args.source_split_id}` test scenes from `Front3d_2988_CFR_Only`.

## Current State

- Target scene count: {report["scene_count"]}
- Completed scenes: {len(scene_summaries)}
- Missing scenes: 0
- Failed scenes: 0
- Interrupted temporary files: {tmp_count}
- Source links: {report["source_num_links"]}
- Noisy links: {report["num_links"]}
- Noise SNR count: {len(args.snr_db)}

## Files

- Build script: `scripts/build_front3d_noise_subset.py`
- Status CSV: `build_status.csv`
- Manifest: `manifest.json`
- Link index: `link_index.pt`

## Resume Command

```bash
python scripts/build_front3d_noise_subset.py --source-root {source_root} --output-root {output_root} --source-split-id {args.source_split_id} --single-scene-split-id {args.single_scene_split_id} --snr-db {" ".join(str(float(v)) for v in args.snr_db)} --seed {args.seed} --chunk-links {args.chunk_links}
```
"""
    (output_root / "BUILD_STATUS.md").write_text(text, encoding="utf-8")


def write_split_readme(path: Path, summary: str) -> None:
    path.write_text(f"# Noise Sub Test Set Split\n\n{summary}\n\nGenerated at: {now_iso()}\n", encoding="utf-8")


def build_final_report(
    output_root: Path,
    source_root: Path,
    args: argparse.Namespace,
    scene_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "output_root": str(output_root),
        "source_root": str(source_root),
        "scene_count": len(scene_summaries),
        "source_num_links": int(sum(item["source_num_links"] for item in scene_summaries)),
        "num_links": int(sum(item["num_links"] for item in scene_summaries)),
        "snr_db": [float(v) for v in args.snr_db],
        "bytes": int(sum(item["output_bytes"] for item in scene_summaries)),
        "manifest": str(output_root / "manifest.json"),
    }


def write_checksums(root: Path) -> None:
    rows = []
    for path in sorted(root.iterdir()):
        if path.name == "checksums.sha256" or not path.is_file():
            continue
        rows.append(f"{sha256_file(path)}  {path.name}")
    (root / "checksums.sha256").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
