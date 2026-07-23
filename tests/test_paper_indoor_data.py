from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from evaluate_mpri import write_metrics_by_snr
from scripts.convert_paper_indoor_data import convert_dataset, load_config
from src.data import CFRDataset, split_indices_by_group


def _write_source(path: Path, *, offset: float, labels: np.ndarray) -> None:
    samples = labels.shape[0]
    values = np.arange(samples * 2 * 3 * 4, dtype=np.float64).reshape(samples, 2, 3, 4)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=values + offset)
        group = handle.create_group("labels")
        group.create_dataset("position", data=labels)


def _write_config(path: Path, source_root: Path, output_path: Path) -> None:
    payload = {
        "dataset": {
            "root": str(source_root),
            "sources": [
                {"file": "snr50.mat", "snr_db": 50},
                {"file": "snr20.mat", "snr_db": 20},
                {"file": "snr10.mat", "snr_db": 10},
            ],
        },
        "expected": {
            "positions": 5,
            "gNBs": 2,
            "mimo_channels": 3,
            "subcarriers": 4,
            "label_dims": 3,
        },
        "output": {
            "path": str(output_path),
            "compression": "none",
            "chunk_samples": 2,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_paper_indoor_converter_preserves_nchw_and_groups_snr_variants(tmp_path: Path) -> None:
    labels = np.stack(
        [
            np.arange(5, dtype=np.float64),
            np.arange(5, dtype=np.float64) + 10,
            np.full(5, 1.2, dtype=np.float64),
        ],
        axis=1,
    )
    _write_source(tmp_path / "snr50.mat", offset=0.0, labels=labels)
    _write_source(tmp_path / "snr20.mat", offset=1000.0, labels=labels)
    _write_source(tmp_path / "snr10.mat", offset=2000.0, labels=labels)
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "converted.h5"
    _write_config(config_path, tmp_path, output_path)

    cfg = load_config(config_path)
    converted, summary = convert_dataset(cfg)

    assert converted == output_path
    assert summary["output_feature_shape_whcn"] == [3, 4, 2, 15]
    dataset = CFRDataset(converted)
    assert tuple(dataset.features_tensor().shape) == (15, 2, 4, 3)
    np.testing.assert_allclose(
        dataset.features_tensor()[0].numpy(),
        np.transpose(np.arange(24).reshape(2, 3, 4), (0, 2, 1)),
    )
    np.testing.assert_allclose(
        dataset.features_tensor()[5].numpy(),
        np.transpose(np.arange(24).reshape(2, 3, 4), (0, 2, 1)) + 1000.0,
    )
    assert dataset.metadata_array("snr_db").astype(int).tolist() == [50] * 5 + [20] * 5 + [10] * 5
    assert dataset.metadata_array("global_mobile_id").tolist() == list(range(5)) * 3

    splits = split_indices_by_group(dataset.group_ids("ue_level"), 0.8, 0.0, 0.2, seed=2026)
    train_groups = set(dataset.group_ids("ue_level")[splits["train"]].tolist())
    test_groups = set(dataset.group_ids("ue_level")[splits["test"]].tolist())
    assert train_groups.isdisjoint(test_groups)
    assert len(splits["train"]) == 12
    assert len(splits["test"]) == 3


def test_paper_indoor_converter_rejects_mismatched_labels(tmp_path: Path) -> None:
    labels = np.zeros((5, 3), dtype=np.float64)
    _write_source(tmp_path / "snr50.mat", offset=0.0, labels=labels)
    _write_source(tmp_path / "snr20.mat", offset=1.0, labels=labels)
    changed = labels.copy()
    changed[0, 0] = 1.0
    _write_source(tmp_path / "snr10.mat", offset=2.0, labels=changed)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, tmp_path, tmp_path / "converted.h5")

    with pytest.raises(ValueError, match="Position labels differ"):
        convert_dataset(load_config(config_path))


def test_paper_indoor_snr_metrics_are_written_per_variant(tmp_path: Path) -> None:
    path = tmp_path / "metrics_by_snr.csv"
    snr_db = np.asarray([50, 20, 10, 50, 20, 10], dtype=np.float32)
    indices = [0, 1, 2, 3, 4, 5]
    errors = np.asarray([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], dtype=np.float32)

    write_metrics_by_snr(path, snr_db, indices, errors)

    rows = path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 4
    assert rows[1].startswith("50.0,")
    assert rows[2].startswith("20.0,")
    assert rows[3].startswith("10.0,")
