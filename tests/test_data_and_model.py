from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.io import savemat

from scripts.convert_cfr_to_csi import build_dataset
from src.data import CFRDataset, split_indices_by_group
from src.model import MPRINet


def _write_front3d_scene(path: Path) -> None:
    ue_ids = np.repeat(np.arange(6, dtype=np.int64), 2)
    bs_ids = np.tile(np.arange(2, dtype=np.int64), 6)
    n = ue_ids.size
    rng = np.random.default_rng(7)
    cfr = (
        rng.standard_normal((n, 4, 2, 32), dtype=np.float32)
        + 1j * rng.standard_normal((n, 4, 2, 32), dtype=np.float32)
    ).astype(np.complex64)
    target_xy = np.stack([ue_ids.astype(np.float32), ue_ids.astype(np.float32) + 10.0], axis=1)
    with h5py.File(path, "w") as f:
        f.create_dataset("cfr", data=cfr)
        f.create_dataset("target_xy", data=target_xy)
        f.create_dataset("ue_id", data=ue_ids)
        f.create_dataset("bs_id", data=bs_ids)
        f.create_dataset("valid_mask", data=np.ones(n, dtype=bool))
        scene = f.create_group("scene")
        scene.create_dataset("scene_id", data=np.bytes_("front3d_unit"))
        link = f.create_group("link")
        link.create_dataset("tx_role", data=np.bytes_("ue"))
        link.create_dataset("rx_role", data=np.bytes_("bs"))
        link.create_dataset("phy_link_direction", data=np.bytes_("uplink"))


def _front3d_cfg(sample_limit: int | None = None) -> dict:
    return {
        "dataset": {"format": "front3d", "root": "", "scene_id": "front3d_unit"},
        "label_dims": 2,
        "representation": "log_abs",
        "eps": 1.0e-8,
        "sample_limit": sample_limit,
        "subcarriers": {"policy": "uniform", "count": 16},
        "augmentation": {"enabled": True, "snr_db": [50.0, 20.0, 10.0], "include_clean": False, "seed": 11},
        "front3d": {"max_bs": 2},
    }


def test_front3d_converter_groups_links_by_ue(tmp_path: Path) -> None:
    scene = tmp_path / "front3d_unit.h5"
    _write_front3d_scene(scene)
    cfg = _front3d_cfg(sample_limit=4)
    data = build_dataset([scene], cfg, require_all_anchors=True)
    assert data["features"].shape == (8, 16, 2, 12)
    assert data["labels"].shape == (12, 2)
    assert data["global_mobile_id"].tolist()[:3] == [0, 0, 0]
    assert sorted(set(data["snr_db"].astype(float).tolist())) == [10.0, 20.0, 50.0]


def _write_deepmimo(root: Path, key: str = "t001_tx000_r000") -> None:
    root.mkdir(parents=True, exist_ok=True)
    n, paths = 10, 6
    rx_pos = np.stack([np.arange(n), np.arange(n) + 100.0, np.full(n, 2.0)], axis=1).astype(np.float32)
    tx_pos = np.asarray([[0.0, 0.0, 6.0]], dtype=np.float32)
    power = -60.0 - np.arange(n * paths, dtype=np.float32).reshape(n, paths) * 0.1
    phase = np.linspace(-180.0, 180.0, n * paths, dtype=np.float32).reshape(n, paths)
    delay = np.full((n, paths), 1.0e-7, dtype=np.float32)
    for field, value in {
        "rx_pos": rx_pos,
        "tx_pos": tx_pos,
        "power": power,
        "phase": phase,
        "delay": delay,
    }.items():
        savemat(root / f"{field}_{key}.mat", {field: value})


def test_deepmimo_converter_synthesizes_virtual_cfr(tmp_path: Path) -> None:
    key = "t001_tx000_r000"
    _write_deepmimo(tmp_path, key)
    cfg = {
        "dataset": {"format": "deepmimo", "root": str(tmp_path), "key": key},
        "label_dims": 2,
        "representation": "amp_phase",
        "eps": 1.0e-8,
        "sample_limit": 5,
        "subcarriers": {"policy": "uniform", "count": 16},
        "augmentation": {"enabled": True, "snr_db": [50.0], "include_clean": True, "seed": 11},
        "deepmimo": {"path_count": 4, "bandwidth_hz": 100_000_000.0},
    }
    data = build_dataset([tmp_path], cfg, require_all_anchors=False)
    assert data["features"].shape == (8, 16, 1, 10)
    np.testing.assert_allclose(data["labels"][0], [0.0, 100.0])
    assert data["global_mobile_id"].tolist()[:2] == [0, 0]


def test_dataset_and_ue_split(tmp_path: Path) -> None:
    scene = tmp_path / "front3d_unit.h5"
    _write_front3d_scene(scene)
    cfg = _front3d_cfg(sample_limit=4)
    data = build_dataset([scene], cfg, require_all_anchors=True)
    out = tmp_path / "processed.h5"
    with h5py.File(out, "w") as f:
        f.create_dataset("features", data=data["features"])
        labels = f.create_group("labels")
        labels.create_dataset("position", data=data["labels"])
        meta = f.create_group("metadata")
        meta.create_dataset("global_mobile_id", data=data["global_mobile_id"])
    ds = CFRDataset(out, preload=True)
    groups = ds.group_ids("ue_level")
    splits = split_indices_by_group(groups, 0.8, 0.0, 0.2, seed=7)
    assert len(splits["val"]) == 0
    assigned = set(splits["train"]) | set(splits["test"])
    assert assigned == set(range(len(ds)))


def test_mpri_outputs_2d() -> None:
    model = MPRINet(in_channels=2, output_dim=2)
    y = model(torch.randn(3, 2, 32, 16))
    assert tuple(y.shape) == (3, 2)
