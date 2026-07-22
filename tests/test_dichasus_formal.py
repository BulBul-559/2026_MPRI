from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from src.dichasus_formal import run_dichasus_formal


def _write_scene(path: Path, run_id: int, count: int) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["contract_version"] = "dichasus_dcxx_csi_v1"
        handle.create_dataset("frequency/model_window_indices", data=np.delete(np.arange(769), 384))
        rng = np.random.default_rng(run_id)
        cfr = (
            rng.standard_normal((count, 32, 1, 769))
            + 1j * rng.standard_normal((count, 32, 1, 769))
        ).astype(np.complex64)
        handle.create_dataset("cfr", data=cfr)
        handle.create_dataset(
            "target_xy", data=np.stack([np.linspace(0, 1, count), np.zeros(count)], axis=1)
        )


def _manifest(path: Path, h5_path: Path, run_id: int, rows: list[int]) -> None:
    payload = {
        "source_h5": {"path": str(h5_path)},
        "rows": [
            {
                "sample_identity": f"dichasus:dc{run_id}:measurement={row}:bs=0",
                "scene_key": f"dichasus_dcxx_dc{run_id}",
                "run_id": run_id,
                "measurement_id": row,
                "bs_id": 0,
                "h5_row": row,
                "link_id": row,
            }
            for row in rows
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_formal_dichasus_mpri_writes_per_link_and_cost_outputs(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    support_h5 = tmp_path / "dc40.h5"
    _write_scene(support_h5, 40, 8)
    _manifest(manifests / "support_n0008.json", support_h5, 40, list(range(8)))
    names = {}
    for run_id in (41, 42, 43, 44, 30, 31, 32, 33, 34):
        h5_path = tmp_path / f"dc{run_id}.h5"
        _write_scene(h5_path, run_id, 2)
        name = ("validation" if run_id == 41 else "evaluation") + f"_dc{run_id}_bs0_full"
        _manifest(manifests / f"{name}.json", h5_path, run_id, [0, 1])
        names[f"dc{run_id}"] = name
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps({"stage2": {"single_bs_manifests": names}}), encoding="utf-8")

    output = run_dichasus_formal(
        protocol_path=protocol_path,
        support_size=8,
        output_dir=tmp_path / "output",
        device="cpu",
        epochs=1,
        batch_size=24,
        eval_batch_size=2,
        subcarrier_count=8,
        max_query_links=1,
    )

    complete = json.loads((output / "complete.json").read_text(encoding="utf-8"))
    assert complete["status"] == "completed"
    assert complete["prediction_count"] == 9
    assert (output / "checkpoints/best.pt").exists()
    assert "sample_identity" in (output / "predictions.csv").read_text(encoding="utf-8")
    cost = json.loads((output / "cost.json").read_text(encoding="utf-8"))
    assert cost["target_label_link_count"] == 8
    assert cost["optimizer_steps"] == 1
