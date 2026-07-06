#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.convert_cfr_to_csi import build_dataset, input_files, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Front3D/DeepMIMO input data for MPRI conversion.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    files = input_files(cfg)
    report = {
        "dataset": cfg["dataset"],
        "input_count": len(files),
        "sample_limit": cfg.get("sample_limit"),
        "representation": cfg["representation"],
        "subcarriers": cfg["subcarriers"],
        "augmentation": cfg["augmentation"],
    }
    if cfg["dataset"]["format"] == "front3d":
        with h5py.File(files[0], "r") as f:
            bs = np.asarray(f["bs_id"] if "bs_id" in f else np.zeros(f["cfr"].shape[0], dtype=np.int64))
            ue = np.asarray(f["ue_id"] if "ue_id" in f else np.arange(f["cfr"].shape[0], dtype=np.int64))
            report.update(
                {
                    "schema": "front3d_cfr_only",
                    "first_file": str(files[0]),
                    "cfr_shape": [int(v) for v in f["cfr"].shape],
                    "target_xy_shape": [int(v) for v in f["target_xy"].shape],
                    "unique_ue": int(np.unique(ue).size),
                    "unique_bs": int(np.unique(bs).size),
                    "valid_links": int(np.asarray(f["valid_mask"] if "valid_mask" in f else np.ones(f["cfr"].shape[0])).sum()),
                }
            )
    else:
        data = build_dataset(files, cfg, require_all_anchors=False)
        report.update(
            {
                "schema": "deepmimo_offline_cfr_hdf5",
                "root": str(files[0]),
                "converted_feature_shape_whcn": [int(v) for v in data["features"].shape],
                "converted_label_shape": [int(v) for v in data["labels"].shape],
                "unique_mobile": int(np.unique(data["global_mobile_id"]).size),
                "anchor_count": int(data["features"].shape[2]),
            }
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
