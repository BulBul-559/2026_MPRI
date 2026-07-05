#!/usr/bin/env python3
"""Create a single-anchor MPRI HDF5 dataset from a processed multi-anchor dataset."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slice a processed MPRI HDF5 to selected anchor channels.")
    parser.add_argument("--input", type=Path, required=True, help="Processed MPRI HDF5 input.")
    parser.add_argument("--output", type=Path, required=True, help="Output HDF5 path.")
    parser.add_argument("--label-json", type=Path, default=None, help="Scene label JSON with bs_points labels.")
    parser.add_argument("--anchor-label", type=str, default=None, help="Anchor label to select, e.g. BS_CENTER.")
    parser.add_argument("--anchor-index", type=int, default=None, help="Anchor index to select if no label is used.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_anchor_labels(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels: list[str] = []
    for item in payload.get("bs_points", []):
        labels.append(str(item.get("label", "")))
    if not labels:
        raise ValueError(f"No bs_points labels found in {path}")
    return labels


def selected_anchor(args: argparse.Namespace, labels: list[str] | None) -> tuple[int, str]:
    if args.anchor_label is not None:
        if labels is None:
            raise ValueError("--anchor-label requires --label-json")
        try:
            return labels.index(args.anchor_label), args.anchor_label
        except ValueError as exc:
            raise ValueError(f"Anchor label not found: {args.anchor_label}; available={labels}") from exc
    if args.anchor_index is None:
        raise ValueError("Provide --anchor-label or --anchor-index")
    label = labels[args.anchor_index] if labels is not None and args.anchor_index < len(labels) else f"anchor_{args.anchor_index}"
    return int(args.anchor_index), label


def copy_dataset(out_group: h5py.Group, name: str, value: h5py.Dataset, data: np.ndarray | None = None) -> None:
    kwargs: dict[str, Any] = {}
    if value.compression is not None:
        kwargs["compression"] = value.compression
        if value.compression_opts is not None:
            kwargs["compression_opts"] = value.compression_opts
    out_group.create_dataset(name, data=np.asarray(value) if data is None else data, dtype=value.dtype, **kwargs)


def copy_attrs(src: h5py.Group | h5py.Dataset, dst: h5py.Group | h5py.Dataset) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def slice_file(input_path: Path, output_path: Path, anchor_index: int, anchor_label: str, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    str_dtype = h5py.string_dtype("utf-8")
    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        features = np.asarray(src["features"])
        if features.ndim != 4:
            raise ValueError(f"Expected features shape (W,H,C,N), got {features.shape}")
        if not 0 <= anchor_index < features.shape[2]:
            raise ValueError(f"anchor_index={anchor_index} outside feature anchor axis {features.shape[2]}")
        dst.create_dataset(
            "features",
            data=features[:, :, anchor_index : anchor_index + 1, :],
            dtype=src["features"].dtype,
            compression=src["features"].compression,
            compression_opts=src["features"].compression_opts,
        )

        labels = dst.create_group("labels")
        copy_attrs(src["labels"], labels)
        for name, dataset in src["labels"].items():
            copy_dataset(labels, name, dataset)

        meta = dst.create_group("metadata")
        copy_attrs(src["metadata"], meta)
        for name, dataset in src["metadata"].items():
            if name == "anchor_positions_m":
                data = np.asarray(dataset)[anchor_index : anchor_index + 1]
                meta.create_dataset(name, data=data, dtype=dataset.dtype)
            elif name == "global_anchor_ids":
                data = np.asarray(dataset)[anchor_index : anchor_index + 1]
                meta.create_dataset(name, data=data, dtype=dataset.dtype)
            elif name == "valid_anchor_mask":
                data = np.asarray(dataset)[:, anchor_index : anchor_index + 1]
                meta.create_dataset(name, data=data, dtype=dataset.dtype)
            else:
                copy_dataset(meta, name, dataset)
        meta.create_dataset("anchor_labels", data=np.asarray([anchor_label], dtype=object).astype(str_dtype), dtype=str_dtype)
        meta.attrs["selected_anchor_index"] = int(anchor_index)
        meta.attrs["selected_anchor_label"] = anchor_label

        config = dst.create_group("config")
        if "config" in src:
            copy_attrs(src["config"], config)
        config.attrs["num_anchors"] = 1
        config.attrs["selected_anchor_index"] = int(anchor_index)
        config.attrs["selected_anchor_label"] = anchor_label
        config.attrs["source_dataset"] = str(input_path)
        config.attrs["created_at"] = datetime.now().isoformat(timespec="seconds")

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "selected_anchor_index": anchor_index,
        "selected_anchor_label": anchor_label,
    }
    output_path.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    labels = read_anchor_labels(args.label_json)
    anchor_index, anchor_label = selected_anchor(args, labels)
    slice_file(args.input, args.output, anchor_index, anchor_label, args.overwrite)


if __name__ == "__main__":
    main()
