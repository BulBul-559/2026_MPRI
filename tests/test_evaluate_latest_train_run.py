from pathlib import Path

import pytest

from scripts.evaluate_latest_train_run import latest_complete_run


def test_latest_complete_run_skips_newer_incomplete_directory(tmp_path: Path) -> None:
    complete = tmp_path / "experiment-2026-01-01-00-00-00"
    incomplete = tmp_path / "experiment-2026-01-02-00-00-00"
    (complete / "checkpoints").mkdir(parents=True)
    (incomplete / "checkpoints").mkdir(parents=True)
    (complete / "checkpoints" / "best.pt").touch()
    (complete / "metrics.csv").touch()
    (complete / "config.yaml").touch()
    (incomplete / "config.yaml").touch()

    selected = latest_complete_run(
        tmp_path,
        "experiment",
        ("checkpoints/best.pt", "metrics.csv", "config.yaml"),
    )

    assert selected == complete


def test_latest_complete_run_requires_all_artifacts(tmp_path: Path) -> None:
    (tmp_path / "experiment-2026-01-01-00-00-00").mkdir()

    with pytest.raises(FileNotFoundError, match="No complete run"):
        latest_complete_run(
            tmp_path,
            "experiment",
            ("checkpoints/best.pt", "metrics.csv", "config.yaml"),
        )
