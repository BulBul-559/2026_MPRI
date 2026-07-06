#!/usr/bin/env python3
"""Convert offline CFR HDF5 data into the MPRI CFR-image HDF5 contract."""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "dataset": {
        "format": "front3d",
        "root_env": "BASELINE_FRONT3D_ROOT",
        "root": "../data/Front3d_2988_CFR_Only",
        "scene_id": "front3d_5431",
    },
    "output_dir": "data/processed/front3d_5431",
    "label_dims": 2,
    "representation": "log_abs",
    "eps": 1.0e-8,
    "sample_limit": None,
    "subcarriers": {"policy": "uniform", "count": 193},
    "augmentation": {
        "enabled": False,
        "snr_db": [50.0, 20.0, 10.0],
        "include_clean": False,
        "seed": 2026,
    },
    "front3d": {"bs_ids": None, "max_bs": None},
    "outputs": {
        "strict_name": "csi_strict_log_abs_193.h5",
        "padded_name": "csi_padded_log_abs_193.h5",
    },
    "compression": "lzf",
    "runs": {"root": "runs", "experiment_name": "convert_mpri"},
}
EXPECTED_DEEPMIMO_CFR_AXES = ("link", "rx_ant", "tx_ant", "subcarrier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert offline CFR HDF5 data for MPRI.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed files.")
    return parser.parse_args()


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        user_cfg = yaml.safe_load(handle) or {}
    if not isinstance(user_cfg, dict):
        raise ValueError("Config top level must be a mapping.")
    cfg = deep_update(DEFAULT_CONFIG, _expand_env(user_cfg))
    dataset = cfg.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("dataset section is required.")
    fmt = str(dataset.get("format", "")).strip().lower()
    if fmt not in {"front3d", "deepmimo"}:
        raise ValueError(f"Unsupported dataset.format: {fmt!r}; expected front3d or deepmimo.")
    dataset["format"] = fmt
    dataset.pop("key", None)
    dataset["root"] = str(_resolve_root(dataset))
    cfg["label_dims"] = int(cfg["label_dims"])
    cfg["representation"] = str(cfg["representation"])
    if cfg["representation"] not in {"log_abs", "amp_phase", "real_imag"}:
        raise ValueError(f"Unsupported representation: {cfg['representation']}")
    cfg["eps"] = float(cfg["eps"])
    cfg["compression"] = str(cfg.get("compression", "lzf"))
    if cfg["compression"] not in {"lzf", "gzip", "none"}:
        raise ValueError("compression must be lzf, gzip or none.")
    cfg["subcarriers"]["count"] = int(cfg["subcarriers"]["count"])
    cfg["subcarriers"]["policy"] = str(cfg["subcarriers"].get("policy", "uniform"))
    if cfg["sample_limit"] is not None:
        cfg["sample_limit"] = int(cfg["sample_limit"])
    aug = cfg["augmentation"]
    aug["enabled"] = bool(aug.get("enabled", False))
    aug["include_clean"] = bool(aug.get("include_clean", False))
    aug["seed"] = int(aug.get("seed", 2026))
    aug["snr_db"] = [float(v) for v in aug.get("snr_db", [])]
    if aug["enabled"] and not aug["snr_db"]:
        raise ValueError("augmentation.snr_db must be non-empty when augmentation.enabled is true.")
    return cfg


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def _resolve_root(dataset: dict[str, Any]) -> Path:
    root_env = dataset.get("root_env")
    root = os.environ.get(str(root_env), "") if root_env else ""
    root = root or str(dataset.get("root", "")).strip()
    if not root:
        raise ValueError(f"dataset.root or environment variable {root_env!r} is required.")
    return Path(root)


def make_run_dir(cfg: dict[str, Any]) -> Path:
    root = Path(cfg.get("runs", {}).get("root", "runs")) / "convert"
    root.mkdir(parents=True, exist_ok=True)
    name = str(cfg.get("runs", {}).get("experiment_name", "convert_mpri"))
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir = root / f"{name}-{stamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{name}-{stamp}-{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def input_files(cfg: dict[str, Any]) -> list[Path]:
    dataset = cfg["dataset"]
    root = Path(dataset["root"])
    if dataset["format"] in {"front3d", "deepmimo"}:
        scene_id = dataset.get("scene_id")
        if scene_id:
            path = root / "scenes" / f"{scene_id}.h5"
            if not path.exists():
                raise FileNotFoundError(f"CFR scene not found: {path}")
            return [path]
        if root.is_file():
            return [root]
        files = sorted((root / "scenes").glob("*.h5")) if (root / "scenes").is_dir() else sorted(root.glob("*.h5"))
        if not files:
            raise FileNotFoundError(f"No CFR .h5 files found under {root}")
        return files
    return [root]


def choose_subcarriers(num_subcarriers: int, policy: str, count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("subcarriers.count must be positive.")
    if count > num_subcarriers:
        raise ValueError(f"subcarriers.count={count} exceeds available subcarriers {num_subcarriers}.")
    if policy == "uniform":
        return np.round(np.linspace(0, num_subcarriers - 1, count)).astype(np.int64)
    if policy == "first":
        return np.arange(count, dtype=np.int64)
    raise ValueError(f"Unsupported subcarrier policy: {policy}")


def _read_text_dataset(f: h5py.File, key: str, default: str = "") -> str:
    if key not in f:
        return default
    value = f[key][()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8")
    return str(value)


def _contract_version(f: h5py.File) -> str:
    value = f.attrs.get("contract_version", "")
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_cfr_axes(f: h5py.File) -> tuple[str, ...] | None:
    raw: Any
    if "cfr_axes" in f:
        raw = _read_text_dataset(f, "cfr_axes")
    elif "cfr_axes" in f.attrs:
        raw = f.attrs["cfr_axes"]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
    else:
        return None
    if isinstance(raw, (list, tuple, np.ndarray)):
        axes = [str(x) for x in raw]
    else:
        axes = json.loads(str(raw))
    return tuple(str(axis) for axis in axes)


def validate_deepmimo_cfr_axes(f: h5py.File, path: Path) -> None:
    if _contract_version(f) != "deepmimo_cfr_v1":
        return
    axes = _read_cfr_axes(f)
    if axes != EXPECTED_DEEPMIMO_CFR_AXES:
        raise ValueError(f"{path} has cfr_axes={axes}; expected {EXPECTED_DEEPMIMO_CFR_AXES}")


def augmentation_levels(cfg: dict[str, Any]) -> list[float | None]:
    aug = cfg["augmentation"]
    if not aug["enabled"]:
        return [None]
    levels: list[float | None] = []
    if aug["include_clean"]:
        levels.append(None)
    levels.extend(float(v) for v in aug["snr_db"])
    return levels


def add_awgn(x: np.ndarray, snr_db: float | None, rng: np.random.Generator) -> np.ndarray:
    if snr_db is None:
        return x
    signal_power = float(np.nanmean(np.abs(x) ** 2))
    if not np.isfinite(signal_power) or signal_power <= 0.0:
        return x
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(x.shape, dtype=np.float32) + 1j * rng.standard_normal(x.shape, dtype=np.float32)
    )
    return (x + noise.astype(np.complex64)).astype(np.complex64)


def complex_to_feature(x: np.ndarray, indices: np.ndarray, representation: str, eps: float) -> np.ndarray:
    selected = x[..., indices]
    flat = selected.reshape(int(np.prod(selected.shape[:-1])), selected.shape[-1])
    if representation == "log_abs":
        return (20.0 * np.log10(np.abs(flat) + eps)).astype(np.float32)
    if representation == "amp_phase":
        out = np.empty((flat.shape[0] * 2, flat.shape[1]), dtype=np.float32)
        out[0::2] = np.abs(flat)
        out[1::2] = np.angle(flat)
        return out
    if representation == "real_imag":
        out = np.empty((flat.shape[0] * 2, flat.shape[1]), dtype=np.float32)
        out[0::2] = flat.real
        out[1::2] = flat.imag
        return out
    raise ValueError(f"Unsupported representation: {representation}")


def build_dataset(files: list[Path], cfg: dict[str, Any], require_all_anchors: bool) -> dict[str, np.ndarray]:
    fmt = cfg["dataset"]["format"]
    if fmt in {"front3d", "deepmimo"}:
        return build_cfr_dataset(files, cfg, require_all_anchors)
    raise ValueError(f"Unsupported dataset format: {fmt}")


def build_cfr_dataset(files: list[Path], cfg: dict[str, Any], require_all_anchors: bool) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(cfg["augmentation"]["seed"]))
    levels = augmentation_levels(cfg)
    features_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    labels3d_parts: list[np.ndarray] = []
    global_mobile_parts: list[np.ndarray] = []
    local_mobile_parts: list[np.ndarray] = []
    shard_parts: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    source_file_parts: list[np.ndarray] = []
    snr_parts: list[np.ndarray] = []
    subcarrier_indices: np.ndarray | None = None
    selected_bs: np.ndarray | None = None

    for shard_i, path in enumerate(files):
        with h5py.File(path, "r") as f:
            if "cfr" not in f or "target_xy" not in f:
                raise ValueError(f"{path} is not an offline CFR scene file.")
            validate_deepmimo_cfr_axes(f, path)
            cfr = np.asarray(f["cfr"])
            labels = np.asarray(f["target_xy"], dtype=np.float32)
            ue_ids = np.asarray(f["ue_id"] if "ue_id" in f else np.arange(cfr.shape[0]), dtype=np.int64)
            bs_ids = np.asarray(f["bs_id"] if "bs_id" in f else np.zeros(cfr.shape[0], dtype=np.int64), dtype=np.int64)
            valid = np.asarray(f["valid_mask"] if "valid_mask" in f else np.ones(cfr.shape[0], dtype=bool), dtype=bool)
            noise_level = np.asarray(f["noise/noise_level_index"], dtype=np.int64) if "noise/noise_level_index" in f else None
            source_snr = np.asarray(f["noise/snr_db"], dtype=np.float32) if "noise/snr_db" in f else None
            if subcarrier_indices is None:
                subcarrier_indices = choose_subcarriers(cfr.shape[-1], cfg["subcarriers"]["policy"], cfg["subcarriers"]["count"])
            if selected_bs is None:
                selected_bs = select_cfr_bs(bs_ids, cfg)
            rows: list[tuple[int, dict[int, int], np.ndarray, float | None]] = cfr_sample_rows(
                labels,
                ue_ids,
                bs_ids,
                valid,
                selected_bs,
                cfg,
                require_all_anchors=require_all_anchors,
                noise_level=noise_level,
                source_snr=source_snr,
            )
            if not rows:
                continue

            sample0 = complex_to_feature(
                cfr[next(iter(rows[0][1].values()))], subcarrier_indices, cfg["representation"], cfg["eps"]
            )
            out_n = len(rows) * len(levels)
            sample_features = np.zeros((sample0.shape[0], subcarrier_indices.size, selected_bs.size, out_n), dtype=np.float32)
            sample_labels = np.zeros((out_n, cfg["label_dims"]), dtype=np.float32)
            sample_labels3d = np.zeros((out_n, 3), dtype=np.float32)
            sample_valid = np.zeros((out_n, selected_bs.size), dtype=np.uint8)
            sample_global = np.zeros(out_n, dtype=np.int64)
            sample_local = np.zeros(out_n, dtype=np.int64)
            sample_snr = np.full(out_n, np.nan, dtype=np.float32)
            out_i = 0
            for local_i, (ue, present, label, row_snr) in enumerate(rows):
                for snr_db in levels:
                    for anchor_i, bs in enumerate(selected_bs.tolist()):
                        idx = present.get(int(bs))
                        if idx is None:
                            continue
                        noisy = add_awgn(cfr[idx], snr_db, rng)
                        sample_features[:, :, anchor_i, out_i] = complex_to_feature(
                            noisy, subcarrier_indices, cfg["representation"], cfg["eps"]
                        )
                        sample_valid[out_i, anchor_i] = 1
                    sample_labels[out_i] = label
                    sample_labels3d[out_i, : cfg["label_dims"]] = label
                    sample_global[out_i] = int(ue)
                    sample_local[out_i] = int(local_i)
                    if row_snr is not None and np.isfinite(row_snr):
                        sample_snr[out_i] = float(row_snr)
                    elif snr_db is not None:
                        sample_snr[out_i] = float(snr_db)
                    out_i += 1

            features_parts.append(sample_features)
            labels_parts.append(sample_labels)
            labels3d_parts.append(sample_labels3d)
            global_mobile_parts.append(sample_global)
            local_mobile_parts.append(sample_local)
            shard_parts.append(np.full(out_n, shard_i, dtype=np.int64))
            valid_parts.append(sample_valid)
            source_file_parts.append(np.asarray([path.name] * out_n, dtype=object))
            snr_parts.append(sample_snr)

    if not features_parts or subcarrier_indices is None or selected_bs is None:
        raise ValueError("No CFR samples were converted.")
    return pack_output(
        cfg,
        features_parts,
        labels_parts,
        labels3d_parts,
        global_mobile_parts,
        local_mobile_parts,
        shard_parts,
        valid_parts,
        source_file_parts,
        snr_parts,
        anchor_ids=selected_bs,
        anchor_positions=np.zeros((selected_bs.size, 3), dtype=np.float32),
        subcarrier_indices=subcarrier_indices,
        layout={
            "localized_side": "link",
            "anchor_side": "bs",
            "tx_role": "ue",
            "rx_role": "bs",
            "phy_link_direction": "uplink",
        },
    )


def cfr_sample_rows(
    labels: np.ndarray,
    ue_ids: np.ndarray,
    bs_ids: np.ndarray,
    valid: np.ndarray,
    selected_bs: np.ndarray,
    cfg: dict[str, Any],
    *,
    require_all_anchors: bool,
    noise_level: np.ndarray | None = None,
    source_snr: np.ndarray | None = None,
) -> list[tuple[int, dict[int, int], np.ndarray, float | None]]:
    if noise_level is None:
        link_map = {(int(ue), int(bs)): i for i, (ue, bs) in enumerate(zip(ue_ids.tolist(), bs_ids.tolist()))}
        ordered_ues = list(dict.fromkeys(int(v) for v in ue_ids.tolist()))
        if cfg["sample_limit"] is not None:
            ordered_ues = ordered_ues[: int(cfg["sample_limit"])]
        rows = []
        for ue in ordered_ues:
            present = {int(bs): link_map[(ue, int(bs))] for bs in selected_bs.tolist() if (ue, int(bs)) in link_map}
            present = {bs: idx for bs, idx in present.items() if bool(valid[idx])}
            append_cfr_row(
                rows,
                ue,
                present,
                labels,
                selected_bs,
                require_all_anchors,
                label_dims=int(cfg["label_dims"]),
                source_snr=None,
            )
        return rows

    link_map = {
        (int(ue), int(level), int(bs)): i
        for i, (ue, level, bs) in enumerate(zip(ue_ids.tolist(), noise_level.tolist(), bs_ids.tolist()))
    }
    ordered_pairs = list(dict.fromkeys((int(ue), int(level)) for ue, level in zip(ue_ids.tolist(), noise_level.tolist())))
    if cfg["sample_limit"] is not None:
        ordered_pairs = ordered_pairs[: int(cfg["sample_limit"])]
    rows = []
    for ue, level in ordered_pairs:
        present = {
            int(bs): link_map[(ue, level, int(bs))]
            for bs in selected_bs.tolist()
            if (ue, level, int(bs)) in link_map
        }
        present = {bs: idx for bs, idx in present.items() if bool(valid[idx])}
        row_snr = None
        if source_snr is not None and present:
            row_snr = float(source_snr[next(iter(present.values()))])
        append_cfr_row(
            rows,
            ue,
            present,
            labels,
            selected_bs,
            require_all_anchors,
            label_dims=int(cfg["label_dims"]),
            source_snr=row_snr,
        )
    return rows


def append_cfr_row(
    rows: list[tuple[int, dict[int, int], np.ndarray, float | None]],
    ue: int,
    present: dict[int, int],
    labels: np.ndarray,
    selected_bs: np.ndarray,
    require_all_anchors: bool,
    *,
    label_dims: int,
    source_snr: float | None,
) -> None:
    if require_all_anchors and len(present) != int(selected_bs.size):
        return
    if not present:
        return
    first_idx = next(iter(present.values()))
    rows.append((ue, present, labels[first_idx, :label_dims], source_snr))


def select_cfr_bs(bs_ids: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    requested = cfg.get("front3d", {}).get("bs_ids")
    if requested:
        values = np.asarray([int(v) for v in requested], dtype=np.int64)
    else:
        values = np.unique(bs_ids.astype(np.int64))
    max_bs = cfg.get("front3d", {}).get("max_bs")
    if max_bs is not None:
        values = values[: int(max_bs)]
    if values.size == 0:
        raise ValueError("No BS ids selected.")
    return values


def pack_output(
    cfg: dict[str, Any],
    features_parts: list[np.ndarray],
    labels_parts: list[np.ndarray],
    labels3d_parts: list[np.ndarray],
    global_mobile_parts: list[np.ndarray],
    local_mobile_parts: list[np.ndarray],
    shard_parts: list[np.ndarray],
    valid_parts: list[np.ndarray],
    source_file_parts: list[np.ndarray],
    snr_parts: list[np.ndarray],
    *,
    anchor_ids: np.ndarray,
    anchor_positions: np.ndarray,
    subcarrier_indices: np.ndarray,
    layout: dict[str, str],
) -> dict[str, np.ndarray]:
    global_mobile = np.concatenate(global_mobile_parts, axis=0)
    local_mobile = np.concatenate(local_mobile_parts, axis=0)
    return {
        "features": np.concatenate(features_parts, axis=3),
        "labels": np.concatenate(labels_parts, axis=0),
        "labels3d": np.concatenate(labels3d_parts, axis=0),
        "global_mobile_id": global_mobile,
        "local_mobile_id": local_mobile,
        "global_tx_id": global_mobile,
        "local_tx_id": local_mobile,
        "shard_index": np.concatenate(shard_parts, axis=0),
        "valid_anchor_mask": np.concatenate(valid_parts, axis=0),
        "source_file": np.concatenate(source_file_parts, axis=0),
        "snr_db": np.concatenate(snr_parts, axis=0),
        "anchor_positions_m": anchor_positions.astype(np.float32),
        "global_anchor_ids": anchor_ids.astype(np.int64),
        "subcarrier_indices": subcarrier_indices.astype(np.int64),
        "layout": layout,
        "dataset_format": np.asarray([cfg["dataset"]["format"]], dtype=object),
    }


def save_output(path: Path, data: dict[str, np.ndarray], cfg: dict[str, Any], mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = None if cfg["compression"] == "none" else cfg["compression"]
    create_kwargs: dict[str, Any] = {"dtype": np.float32}
    if compression is not None:
        create_kwargs["compression"] = compression
        if compression == "gzip":
            create_kwargs["compression_opts"] = 4
    str_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=data["features"], **create_kwargs)
        labels = f.create_group("labels")
        labels.create_dataset("position", data=data["labels"], **create_kwargs)
        labels.create_dataset("position3d", data=data["labels3d"], **create_kwargs)
        meta = f.create_group("metadata")
        for key in [
            "global_tx_id",
            "local_tx_id",
            "global_mobile_id",
            "local_mobile_id",
            "shard_index",
            "valid_anchor_mask",
            "anchor_positions_m",
            "global_anchor_ids",
            "subcarrier_indices",
            "snr_db",
        ]:
            meta.create_dataset(key, data=data[key])
        meta.create_dataset("source_file", data=data["source_file"].astype(str_dtype), dtype=str_dtype)
        config = f.create_group("config")
        layout = data["layout"]
        attrs = {
            "mode": mode,
            "dataset_format": cfg["dataset"]["format"],
            "dataset_root": cfg["dataset"]["root"],
            "representation": cfg["representation"],
            "label_dims": cfg["label_dims"],
            "num_samples": int(data["labels"].shape[0]),
            "num_anchors": int(data["features"].shape[2]),
            "num_subcarriers": int(data["features"].shape[1]),
            "feature_width": int(data["features"].shape[0]),
            "augmentation_enabled": bool(cfg["augmentation"]["enabled"]),
            "augmentation_snr_db": json.dumps(cfg["augmentation"]["snr_db"]),
            "augmentation_note": "AWGN is applied to offline CFR as an adaptation of paper PHY-layer Filter_multipath noise.",
            "localized_side": layout["localized_side"],
            "anchor_side": layout["anchor_side"],
            "tx_role": layout["tx_role"],
            "rx_role": layout["rx_role"],
            "phy_link_direction": layout["phy_link_direction"],
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        for key, value in attrs.items():
            config.attrs[key] = value


def write_metadata_json(path: Path, data: dict[str, np.ndarray], cfg: dict[str, Any], mode: str) -> None:
    payload = {
        "mode": mode,
        "dataset_format": cfg["dataset"]["format"],
        "dataset_root": cfg["dataset"]["root"],
        "dataset_scene": cfg["dataset"].get("scene_id"),
        "representation": cfg["representation"],
        "label_dims": cfg["label_dims"],
        "feature_shape_whcn": [int(v) for v in data["features"].shape],
        "label_shape": [int(v) for v in data["labels"].shape],
        "subcarrier_policy": cfg["subcarriers"]["policy"],
        "subcarrier_count": int(data["subcarrier_indices"].size),
        "sample_count": int(data["labels"].shape[0]),
        "unique_global_mobile_count": int(np.unique(data["global_mobile_id"]).size),
        "global_anchor_ids": data["global_anchor_ids"].astype(int).tolist(),
        "augmentation": cfg["augmentation"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    path.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def convert(cfg: dict[str, Any], overwrite: bool) -> Path:
    files = input_files(cfg)
    output_dir = Path(cfg["output_dir"])
    strict_path = output_dir / cfg["outputs"]["strict_name"]
    padded_path = output_dir / cfg["outputs"]["padded_name"]
    for path in (strict_path, padded_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output file exists: {path}; use --overwrite")

    run_dir = make_run_dir(cfg)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    report: dict[str, Any] = {
        "dataset": cfg["dataset"],
        "input_count": len(files),
        "outputs": {},
    }
    for mode, require_all, path in (
        ("strict", True, strict_path),
        ("padded", False, padded_path),
    ):
        data = build_dataset(files, cfg, require_all_anchors=require_all)
        save_output(path, data, cfg, mode)
        write_metadata_json(path, data, cfg, mode)
        report["outputs"][mode] = {
            "path": str(path),
            "features_shape_whcn": [int(v) for v in data["features"].shape],
            "labels_shape": [int(v) for v in data["labels"].shape],
            "unique_global_mobile_count": int(np.unique(data["global_mobile_id"]).size),
            "num_anchors": int(data["features"].shape[2]),
            "snr_levels": sorted({float(v) for v in data["snr_db"] if np.isfinite(v)}),
        }

    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(run_dir)
    return run_dir


def main() -> None:
    args = parse_args()
    try:
        cfg = load_config(args.config)
        convert(cfg, overwrite=args.overwrite)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
