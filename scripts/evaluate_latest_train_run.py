#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def latest_complete_run(runs_root: Path, experiment_name: str, required: tuple[str, ...]) -> Path:
    candidates = sorted(
        runs_root.glob(f"{experiment_name}-*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if all((candidate / relative).is_file() for relative in required):
            return candidate
    raise FileNotFoundError(
        f"No complete run for {experiment_name!r} under {runs_root}; required={list(required)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the latest complete MPRI training run.")
    parser.add_argument("--train-runs-root", type=Path, required=True)
    parser.add_argument("--train-experiment-name", required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="best.pt")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_run = latest_complete_run(
        args.train_runs_root,
        args.train_experiment_name,
        (f"checkpoints/{args.checkpoint_name}", "metrics.csv", "config.yaml"),
    )
    checkpoint = train_run / "checkpoints" / args.checkpoint_name
    command = [
        sys.executable,
        "evaluate_mpri.py",
        "--config",
        str(args.eval_config),
        "--checkpoint",
        str(checkpoint),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        return completed.returncode

    output_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise RuntimeError("MPRI evaluation produced no stdout containing its run directory")
    eval_run = Path(output_lines[-1])
    if not eval_run.is_absolute():
        eval_run = PROJECT_ROOT / eval_run
    eval_run = eval_run.resolve()
    required_eval = [eval_run / "metrics.csv", eval_run / "predictions.csv", eval_run / "config.yaml"]
    missing = [str(path) for path in required_eval if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Evaluation run is incomplete: {missing}")

    summary = {
        "schema_version": "mpri_train_eval_summary_v1",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "train_run": str(train_run.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "train_metrics": str((train_run / "metrics.csv").resolve()),
        "eval_run": str(eval_run),
        "eval_metrics": str((eval_run / "metrics.csv").resolve()),
        "predictions": str((eval_run / "predictions.csv").resolve()),
        "eval_config": str(args.eval_config.resolve()),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
