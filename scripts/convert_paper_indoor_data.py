#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the original MPRI indoor MATLAB v7.3 files to the project HDF5 contract."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="Replace an existing converted dataset.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate source files without writing output.",
    )
    return parser.parse_args()


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping.")

    dataset = cfg.get("dataset")
    output = cfg.get("output")
    expected = cfg.get("expected", {})
    if not isinstance(dataset, dict) or not isinstance(output, dict) or not isinstance(expected, dict):
        raise ValueError("Config requires mapping sections: dataset, output, expected.")

    root_env = str(dataset.get("root_env", "")).strip()
    root_value = os.environ.get(root_env) if root_env else None
    if not root_value:
        root_value = dataset.get("root")
    if not root_value:
        raise ValueError("dataset.root or dataset.root_env must resolve to a source directory.")
    dataset["root"] = str(_resolve_project_path(str(root_value)))

    sources = dataset.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("dataset.sources must be a non-empty list.")
    normalized_sources: list[dict[str, Any]] = []
    seen_snr: set[float] = set()
    for item in sources:
        if not isinstance(item, dict) or "file" not in item or "snr_db" not in item:
            raise ValueError("Each dataset.sources entry requires file and snr_db.")
        snr_db = float(item["snr_db"])
        if snr_db in seen_snr:
            raise ValueError(f"Duplicate SNR source: {snr_db:g} dB")
        seen_snr.add(snr_db)
        normalized_sources.append({"file": str(item["file"]), "snr_db": snr_db})
    dataset["sources"] = normalized_sources

    if "path" not in output:
        raise ValueError("output.path is required.")
    output["path"] = str(_resolve_project_path(str(output["path"])))
    output["compression"] = str(output.get("compression", "gzip")).lower()
    output["compression_level"] = int(output.get("compression_level", 4))
    output["chunk_samples"] = int(output.get("chunk_samples", 16))
    if output["compression"] not in {"gzip", "lzf", "none"}:
        raise ValueError("output.compression must be one of: gzip, lzf, none.")
    if output["chunk_samples"] <= 0:
        raise ValueError("output.chunk_samples must be positive.")

    expected["positions"] = int(expected.get("positions", 4816))
    expected["gNBs"] = int(expected.get("gNBs", 5))
    expected["mimo_channels"] = int(expected.get("mimo_channels", 16))
    expected["subcarriers"] = int(expected.get("subcarriers", 193))
    expected["label_dims"] = int(expected.get("label_dims", 3))
    cfg["dataset"] = dataset
    cfg["output"] = output
    cfg["expected"] = expected
    return cfg


def inspect_sources(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg["dataset"]["root"])
    expected = cfg["expected"]
    expected_feature_shape = (
        expected["positions"],
        expected["gNBs"],
        expected["mimo_channels"],
        expected["subcarriers"],
    )
    expected_label_shape = (expected["positions"], expected["label_dims"])

    reference_labels: np.ndarray | None = None
    sources: list[dict[str, Any]] = []
    for variant_index, source in enumerate(cfg["dataset"]["sources"]):
        path = root / source["file"]
        if not path.is_file():
            raise FileNotFoundError(f"MPRI source file not found: {path}")
        with h5py.File(path, "r") as handle:
            if "features" not in handle or "labels/position" not in handle:
                raise ValueError(f"Missing features or labels/position in {path}")
            features = handle["features"]
            labels = np.asarray(handle["labels/position"], dtype=np.float64)
            if tuple(features.shape) != expected_feature_shape:
                raise ValueError(
                    f"Unexpected features shape in {path}: {tuple(features.shape)}; "
                    f"expected {expected_feature_shape}"
                )
            if tuple(labels.shape) != expected_label_shape:
                raise ValueError(
                    f"Unexpected labels shape in {path}: {tuple(labels.shape)}; "
                    f"expected {expected_label_shape}"
                )
            if not np.issubdtype(features.dtype, np.floating):
                raise ValueError(f"Expected real-valued CFR images in {path}, got {features.dtype}")
            if not np.isfinite(labels).all():
                raise ValueError(f"Non-finite labels found in {path}")
            if reference_labels is None:
                reference_labels = labels
            elif not np.array_equal(reference_labels, labels):
                max_diff = float(np.max(np.abs(reference_labels - labels)))
                raise ValueError(f"Position labels differ across SNR files; max_abs_diff={max_diff}")
            sources.append(
                {
                    "variant_index": variant_index,
                    "path": str(path.resolve()),
                    "file": path.name,
                    "snr_db": float(source["snr_db"]),
                    "feature_shape_h5py": list(features.shape),
                    "feature_dtype": str(features.dtype),
                    "label_shape": list(labels.shape),
                }
            )

    assert reference_labels is not None
    unique_positions = int(np.unique(reference_labels, axis=0).shape[0])
    if unique_positions != expected["positions"]:
        raise ValueError(
            f"Expected {expected['positions']} unique positions, found {unique_positions}"
        )
    return {
        "schema_version": "mpri_paper_indoor_source_audit_v1",
        "source_root": str(root.resolve()),
        "sources": sources,
        "positions": expected["positions"],
        "snr_variant_count": len(sources),
        "sample_count": expected["positions"] * len(sources),
        "feature_shape_h5py": list(expected_feature_shape),
        "paper_matlab_shape": [
            expected["subcarriers"],
            expected["mimo_channels"],
            expected["gNBs"],
            expected["positions"],
        ],
        "coordinate_min_m": reference_labels.min(axis=0).tolist(),
        "coordinate_max_m": reference_labels.max(axis=0).tolist(),
        "labels_equal_across_snr": True,
    }


def _compression_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    compression = cfg["output"]["compression"]
    if compression == "none":
        return {}
    kwargs: dict[str, Any] = {"compression": compression, "shuffle": True}
    if compression == "gzip":
        kwargs["compression_opts"] = cfg["output"]["compression_level"]
    return kwargs


def convert_dataset(cfg: dict[str, Any], *, force: bool = False) -> tuple[Path, dict[str, Any]]:
    audit = inspect_sources(cfg)
    output_path = Path(cfg["output"]["path"])
    if output_path.exists() and not force:
        raise FileExistsError(f"Output already exists (use --force to replace): {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.unlink(missing_ok=True)

    expected = cfg["expected"]
    positions = int(expected["positions"])
    variants = len(cfg["dataset"]["sources"])
    total_samples = positions * variants
    gnb_count = int(expected["gNBs"])
    mimo_channels = int(expected["mimo_channels"])
    subcarriers = int(expected["subcarriers"])
    chunk_samples = min(int(cfg["output"]["chunk_samples"]), positions)
    compression_kwargs = _compression_kwargs(cfg)

    source_root = Path(cfg["dataset"]["root"])
    source_files = [source_root / source["file"] for source in cfg["dataset"]["sources"]]
    try:
        with h5py.File(tmp_path, "w") as output:
            features_out = output.create_dataset(
                "features",
                shape=(mimo_channels, subcarriers, gnb_count, total_samples),
                dtype=np.float32,
                chunks=(mimo_channels, subcarriers, gnb_count, 1),
                **compression_kwargs,
            )
            labels_group = output.create_group("labels")
            labels_out = labels_group.create_dataset(
                "position",
                shape=(total_samples, expected["label_dims"]),
                dtype=np.float32,
                chunks=(min(1024, total_samples), expected["label_dims"]),
                **compression_kwargs,
            )
            metadata = output.create_group("metadata")
            global_mobile_id = np.tile(np.arange(positions, dtype=np.int64), variants)
            metadata.create_dataset("global_mobile_id", data=global_mobile_id)
            metadata.create_dataset("source_position_index", data=global_mobile_id)
            metadata.create_dataset(
                "source_variant_index",
                data=np.repeat(np.arange(variants, dtype=np.int16), positions),
            )
            metadata.create_dataset(
                "snr_db",
                data=np.concatenate(
                    [
                        np.full(positions, float(source["snr_db"]), dtype=np.float32)
                        for source in cfg["dataset"]["sources"]
                    ]
                ),
            )
            source_file_dtype = h5py.string_dtype(encoding="utf-8")
            metadata.create_dataset(
                "source_file",
                data=np.concatenate(
                    [
                        np.full(positions, source["file"], dtype=object)
                        for source in cfg["dataset"]["sources"]
                    ]
                ),
                dtype=source_file_dtype,
            )
            metadata.create_dataset("global_anchor_ids", data=np.arange(gnb_count, dtype=np.int64))
            metadata.create_dataset(
                "valid_anchor_mask",
                data=np.ones((total_samples, gnb_count), dtype=np.bool_),
                chunks=(min(1024, total_samples), gnb_count),
                compression="gzip",
                compression_opts=1,
                shuffle=True,
            )

            for variant_index, path in enumerate(source_files):
                sample_offset = variant_index * positions
                with h5py.File(path, "r") as source_handle:
                    source_features = source_handle["features"]
                    labels = np.asarray(source_handle["labels/position"], dtype=np.float32)
                    labels_out[sample_offset : sample_offset + positions] = labels
                    for start in range(0, positions, chunk_samples):
                        stop = min(start + chunk_samples, positions)
                        # h5py axes are (sample, gNB, MIMO channel, subcarrier).
                        # Project storage is (W, H, C, N), where H is frequency.
                        block_ngms = np.asarray(source_features[start:stop], dtype=np.float32)
                        block_whcn = np.transpose(block_ngms, (2, 3, 1, 0))
                        features_out[
                            :,
                            :,
                            :,
                            sample_offset + start : sample_offset + stop,
                        ] = block_whcn

            config_group = output.create_group("config")
            config_group.attrs["contract"] = "mpri_cfr_image_v1"
            config_group.attrs["dataset_family"] = "mpri_paper_indoor"
            config_group.attrs["source_format"] = "matlab_v7.3_hdf5"
            config_group.attrs["paper_scenario"] = "scenario_1_indoor_office_hall"
            config_group.attrs["storage_axes"] = json.dumps(
                ["mimo_channel", "subcarrier", "gNB", "sample"]
            )
            config_group.attrs["loader_axes"] = json.dumps(
                ["sample", "gNB", "subcarrier", "mimo_channel"]
            )
            config_group.attrs["snr_db"] = json.dumps(
                [float(source["snr_db"]) for source in cfg["dataset"]["sources"]]
            )
            config_group.attrs["noise_insertion"] = "author_generated_PHY_Filter_multipath"
            config_group.attrs["secondary_noise_augmentation"] = False
            config_group.attrs["source_root"] = str(source_root.resolve())
            config_group.attrs["source_files"] = json.dumps(
                [source["file"] for source in cfg["dataset"]["sources"]]
            )
            config_group.attrs["sample_order"] = "source_variant_major_then_position"
            config_group.attrs["position_grouping"] = "metadata/global_mobile_id"
            output.flush()
        tmp_path.replace(output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    summary = {
        **audit,
        "schema_version": "mpri_paper_indoor_conversion_v1",
        "output": str(output_path.resolve()),
        "output_feature_shape_whcn": [
            mimo_channels,
            subcarriers,
            gnb_count,
            total_samples,
        ],
        "output_label_shape": [total_samples, expected["label_dims"]],
        "output_dtype": "float32",
        "sample_order": "source_variant_major_then_position",
    }
    return output_path, summary


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.check_only:
        report = inspect_sources(cfg)
    else:
        _, report = convert_dataset(cfg, force=args.force)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
