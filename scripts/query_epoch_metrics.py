#!/usr/bin/env python3
"""Query per-epoch MPRI metrics from a training run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_COLUMNS = [
    "epoch",
    "lr",
    "train_mean_err",
    "val_mean_err",
    "test_mean_err",
    "train_rmse",
    "val_rmse",
    "test_rmse",
    "is_best",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query selected epochs from MPRI metrics.csv.")
    parser.add_argument("path", type=Path, help="Path to metrics.csv or a run directory containing it.")
    parser.add_argument("--epochs", type=int, nargs="*", default=None, help="Epoch numbers to print.")
    parser.add_argument(
        "--columns",
        nargs="+",
        default=DEFAULT_COLUMNS,
        help="Metric columns to print. Defaults to mean-error and RMSE columns.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text table.")
    return parser.parse_args()


def metrics_path(path: Path) -> Path:
    if path.is_dir():
        path = path / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"metrics.csv not found: {path}")
    return path


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_rows(rows: list[dict[str, str]], epochs: list[int] | None) -> list[dict[str, str]]:
    if epochs is None:
        return rows
    wanted = {int(epoch) for epoch in epochs}
    return [row for row in rows if int(row["epoch"]) in wanted]


def format_value(value: str) -> str:
    if value == "":
        return ""
    try:
        number = float(value)
    except ValueError:
        return value
    if number != number:
        return "nan"
    if abs(number) >= 1000 or (0 < abs(number) < 1e-4):
        return f"{number:.6g}"
    return f"{number:.6f}".rstrip("0").rstrip(".")


def print_table(rows: list[dict[str, str]], columns: list[str]) -> None:
    rendered: list[dict[str, str]] = []
    for row in rows:
        rendered.append({column: format_value(row.get(column, "")) for column in columns})
    widths = {
        column: max(len(column), *(len(row[column]) for row in rendered)) if rendered else len(column)
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("  ".join("-" * widths[column] for column in columns))
    for row in rendered:
        print("  ".join(row[column].ljust(widths[column]) for column in columns))


def to_jsonable(rows: list[dict[str, str]], columns: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for column in columns:
            value = row.get(column, "")
            try:
                item[column] = int(value) if column == "epoch" else float(value)
            except ValueError:
                item[column] = value
        out.append(item)
    return out


def main() -> None:
    args = parse_args()
    path = metrics_path(args.path)
    rows = select_rows(load_rows(path), args.epochs)
    if args.json:
        print(json.dumps(to_jsonable(rows, args.columns), ensure_ascii=False, indent=2))
    else:
        print_table(rows, args.columns)


if __name__ == "__main__":
    main()
