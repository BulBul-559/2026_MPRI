from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from scripts.convert_cfr_to_csi import build_dataset, input_files, load_config, select_cfr_bs
from src.data import CFRDataset, split_indices_by_group
from src.model import MPRINet, count_parameters
from train_mpri import split_indices_from_plan


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
        f.attrs["cfr_axes"] = '["link", "rx_ant", "tx_ant", "subcarrier"]'
        f.create_dataset("cfr_axes", data=np.bytes_('["link", "rx_ant", "tx_ant", "subcarrier"]'))
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


def _write_noise_status(root: Path, rows: list[tuple[str, str, int]]) -> None:
    lines = [
        "scene_order,scene_id,status,h5_path,tmp_path,size_bytes,source_num_links,num_links,noise_snr_count,error"
    ]
    for order, (scene_id, status, snr_count) in enumerate(rows, start=1):
        h5_path = root / "scenes" / f"{scene_id}.h5"
        lines.append(f"{order},{scene_id},{status},{h5_path.as_posix()},,1,1,3,{snr_count},")
    (root / "build_status.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_noise_scene_file_uses_dedicated_root_env_and_current_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clean_root = tmp_path / "Front3d_2988_CFR_Only"
    noise_root = clean_root / "noise_sub_test_set"
    (clean_root / "scenes").mkdir(parents=True)
    (noise_root / "scenes").mkdir(parents=True)
    split_dir = noise_root / "splits" / "m117_seed2026_scene_level"
    split_dir.mkdir(parents=True)
    (noise_root / "scenes" / "front3d_a.h5").touch()
    (noise_root / "scenes" / "front3d_b.h5").touch()
    (split_dir / "test_scenes.txt").write_text("front3d_a\nfront3d_b\n", encoding="utf-8")
    _write_noise_status(noise_root, [("front3d_a", "completed", 3), ("front3d_b", "completed", 3)])

    monkeypatch.setenv("BASELINE_FRONT3D_ROOT", str(clean_root))
    monkeypatch.setenv("BASELINE_FRONT3D_NOISE_ROOT", str(noise_root))
    cfg_path = tmp_path / "current_noise.yaml"
    cfg_path.write_text(
        f"""
dataset:
  format: front3d
  root_env: BASELINE_FRONT3D_NOISE_ROOT
  root: {clean_root.as_posix()}
  scene_ids_file: splits/m117_seed2026_scene_level/test_scenes.txt
  scene_limit: 1
  require_complete_build: true
  expected_scene_count: 2
  expected_noise_snr_count: 3
output_dir: out
""",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    files = input_files(cfg)
    assert Path(cfg["dataset"]["root"]) == noise_root
    assert files == [noise_root / "scenes" / "front3d_a.h5"]


def test_noise_build_status_rejects_incomplete_requested_scene(tmp_path: Path) -> None:
    noise_root = tmp_path / "noise_sub_test_set"
    (noise_root / "scenes").mkdir(parents=True)
    _write_noise_status(noise_root, [("front3d_bad", "failed", 3)])
    cfg_path = tmp_path / "bad_noise.yaml"
    cfg_path.write_text(
        f"""
dataset:
  format: front3d
  root: {noise_root.as_posix()}
  scene_ids: [front3d_bad]
  require_complete_build: true
  expected_scene_count: 1
  expected_noise_snr_count: 3
output_dir: out
""",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    with pytest.raises(ValueError, match="not complete"):
        input_files(cfg)


def test_front3d_converter_groups_links_by_ue(tmp_path: Path) -> None:
    scene = tmp_path / "front3d_unit.h5"
    _write_front3d_scene(scene)
    cfg = _front3d_cfg(sample_limit=4)
    data = build_dataset([scene], cfg, require_all_anchors=True)
    assert data["features"].shape == (8, 16, 2, 12)
    assert data["labels"].shape == (12, 2)
    assert data["global_mobile_id"].tolist()[:3] == [0, 0, 0]
    assert data["source_link_ids"].shape == (12, 2)
    assert data["source_link_ids"][0].tolist() == [0, 1]
    assert sorted(set(data["snr_db"].astype(float).tolist())) == [10.0, 20.0, 50.0]


def test_front3d_converter_filters_to_m117_link_plan(tmp_path: Path) -> None:
    scene = tmp_path / "front3d_unit.h5"
    _write_front3d_scene(scene)
    plan_dir = tmp_path / "front3d_unit_seed2026_ue_level"
    plan_dir.mkdir()
    torch.save(
        {
            "local_link_index": torch.tensor([0, 1, 2, 3], dtype=torch.long),
            "ue_id": torch.tensor([0, 0, 1, 1], dtype=torch.long),
            "bs_id": torch.tensor([0, 1, 0, 1], dtype=torch.long),
        },
        plan_dir / "train_links.pt",
    )
    torch.save(
        {
            "local_link_index": torch.tensor([4, 5], dtype=torch.long),
            "ue_id": torch.tensor([2, 2], dtype=torch.long),
            "bs_id": torch.tensor([0, 1], dtype=torch.long),
        },
        plan_dir / "val_links.pt",
    )
    torch.save(
        {
            "local_link_index": torch.tensor([6, 7], dtype=torch.long),
            "ue_id": torch.tensor([3, 3], dtype=torch.long),
            "bs_id": torch.tensor([0, 1], dtype=torch.long),
        },
        plan_dir / "test_links.pt",
    )
    cfg = _front3d_cfg()
    cfg["dataset"]["split_plan_dir"] = str(plan_dir)

    data = build_dataset([scene], cfg, require_all_anchors=False)

    assert set(data["global_mobile_id"].astype(int).tolist()) == {0, 1, 2, 3}
    assert 4 not in set(data["global_mobile_id"].astype(int).tolist())
    assert 5 not in set(data["global_mobile_id"].astype(int).tolist())
    assert set(data["source_link_ids"].reshape(-1).astype(int).tolist()) <= {-1, 0, 1, 2, 3, 4, 5, 6, 7}


def test_converter_supports_deterministic_random_bs_selection() -> None:
    bs_ids = np.arange(10, dtype=np.int64)
    cfg = _front3d_cfg()
    cfg["dataset"]["scene_id"] = "front3d_unit"
    cfg["front3d"] = {"bs_count": 4, "bs_selection": "random", "bs_seed": 2026}
    selected = select_cfr_bs(bs_ids, cfg)
    selected_again = select_cfr_bs(bs_ids, cfg)
    assert selected.tolist() == [1, 2, 6, 8]
    assert selected_again.tolist() == selected.tolist()


def test_converter_single_bs_random_selection_is_deterministic() -> None:
    bs_ids = np.asarray([3, 1, 5], dtype=np.int64)
    cfg = _front3d_cfg()
    cfg["dataset"]["scene_id"] = "front3d_unit"
    cfg["front3d"] = {"bs_count": 1, "bs_selection": "random", "bs_seed": 2026}
    selected = select_cfr_bs(bs_ids, cfg)
    selected_again = select_cfr_bs(bs_ids, cfg)
    assert selected.tolist() == selected_again.tolist()
    assert len(selected.tolist()) == 1
    assert selected.tolist()[0] in {1, 3, 5}


def test_converter_uses_bs_selection_contract(tmp_path: Path) -> None:
    contract = tmp_path / "bs_selection_contract.json"
    contract.write_text(
        """
{
  "schema_version": "baseline_bs_selection_contract_v1",
  "contract_id": "bs_selection_v1",
  "scenes": [
    {"scene_id": "front3d_unit", "single_bs_ids": [3], "multi_bs_ids": [1, 4]}
  ]
}
""".strip(),
        encoding="utf-8",
    )
    cfg = _front3d_cfg()
    cfg["dataset"]["scene_id"] = "front3d_unit"
    cfg["front3d"] = {"bs_count": 2, "bs_selection": "random", "bs_seed": 2026, "bs_contract": str(contract)}

    selected = select_cfr_bs(np.arange(6, dtype=np.int64), cfg)

    assert selected.tolist() == [1, 4]


def test_deepmimo_converter_consumes_offline_cfr(tmp_path: Path) -> None:
    scene = tmp_path / "deepmimo_cfr_unit.h5"
    _write_front3d_scene(scene)
    cfg = {
        "dataset": {"format": "deepmimo", "root": str(tmp_path), "scene_id": "deepmimo_cfr_unit"},
        "label_dims": 2,
        "representation": "amp_phase",
        "eps": 1.0e-8,
        "sample_limit": 5,
        "subcarriers": {"policy": "uniform", "count": 16},
        "augmentation": {"enabled": True, "snr_db": [50.0], "include_clean": True, "seed": 11},
        "front3d": {"max_bs": 2},
    }
    data = build_dataset([scene], cfg, require_all_anchors=True)
    assert data["features"].shape == (16, 16, 2, 10)
    np.testing.assert_allclose(data["labels"][0], [0.0, 10.0])
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


def test_mpri_uses_predefined_scene_ue_split_plan(tmp_path: Path) -> None:
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

    plan_dir = tmp_path / "front3d_unit_seed2026_ue_level"
    plan_dir.mkdir()
    torch.save({"ue_id": torch.tensor([0, 1], dtype=torch.long)}, plan_dir / "train_links.pt")
    torch.save({"ue_id": torch.tensor([2], dtype=torch.long)}, plan_dir / "val_links.pt")
    torch.save({"ue_id": torch.tensor([3], dtype=torch.long)}, plan_dir / "test_links.pt")

    ds = CFRDataset(out, preload=True)
    splits = split_indices_from_plan(ds, plan_dir)

    assert len(splits["train"]) == 6
    assert len(splits["val"]) == 3
    assert len(splits["test"]) == 3
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert set(splits["train"]).isdisjoint(splits["test"])


def test_mpri_outputs_2d() -> None:
    model = MPRINet(in_channels=2, output_dim=2)
    y = model(torch.randn(3, 2, 32, 16))
    assert tuple(y.shape) == (3, 2)


def test_mpri_paper_scale_matches_reported_parameter_count() -> None:
    model = MPRINet(in_channels=1, output_dim=3)
    n_params = count_parameters(model)
    assert 2_550_000 <= n_params <= 2_700_000
    y = model(torch.randn(2, 1, 193, 16))
    assert tuple(y.shape) == (2, 3)
