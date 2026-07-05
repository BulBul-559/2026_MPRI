#!/usr/bin/env python3
"""Plot Front3D MPRI test predictions on scene floorplans."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from PIL import Image


SCENE_ROOT = Path("/data/sunmeiyuan/projects/SceneGen/front3d_full_all_6813_20260617_215824")
DEFAULT_SCENES: list[str] = []
DEFAULT_EVAL_RUNS: dict[str, Path] = {}


@dataclass(frozen=True)
class ErrorBin:
    label: str
    lower: float
    upper: float
    color: str

    def contains(self, value: float) -> bool:
        return self.lower <= value < self.upper


ERROR_BINS = [
    ErrorBin("<=1m", 0.0, 1.0, "#35d36b"),
    ErrorBin("1-2m", 1.0, 2.0, "#f1d33f"),
    ErrorBin("2-5m", 2.0, 5.0, "#f29e2e"),
    ErrorBin(">5m", 5.0, float("inf"), "#ef4b4b"),
]


@dataclass(frozen=True)
class PlotCase:
    label: str
    scene: str
    eval_run: Path
    bs_index: int | None = None

    @property
    def slug(self) -> str:
        safe_label = (
            self.label.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("|", "_")
            .replace("__", "_")
        )
        return f"{self.scene}_{safe_label}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MPRI prediction arrows on Front3D floorplans.")
    parser.add_argument("--scene-root", type=Path, default=SCENE_ROOT)
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to runs/figures/mpri_front3d_test_visuals-<timestamp>.",
    )
    parser.add_argument(
        "--use-label-floorplan",
        action="store_true",
        help="Use label_floorplan/label_panel_0p5.png instead of floorplan/floorplan_1p60.png.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="LABEL|SCENE|EVAL_RUN|BS_INDEX",
        help=(
            "Plot one explicit case. May be repeated. BS_INDEX is optional; "
            "when present only that BS is drawn."
        ),
    )
    parser.add_argument(
        "--overview-cols",
        type=int,
        default=None,
        help="Number of columns in the overview grid. Defaults to case count for default plots, 3 for explicit cases.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def parse_cases(args: argparse.Namespace) -> list[PlotCase]:
    if not args.case:
        if not args.scenes:
            raise ValueError("No default MPRI visualization cases are configured; use --case LABEL|SCENE|EVAL_RUN.")
        return [PlotCase(scene, scene, DEFAULT_EVAL_RUNS[scene]) for scene in args.scenes]

    cases: list[PlotCase] = []
    for raw in args.case:
        parts = raw.split("|")
        if len(parts) not in {3, 4}:
            raise ValueError(
                "--case must use LABEL|SCENE|EVAL_RUN or LABEL|SCENE|EVAL_RUN|BS_INDEX"
            )
        label, scene, eval_run = parts[:3]
        bs_index = int(parts[3]) if len(parts) == 4 and parts[3] != "" else None
        cases.append(PlotCase(label=label, scene=scene, eval_run=Path(eval_run), bs_index=bs_index))
    return cases


def read_predictions(eval_run: Path) -> list[dict[str, float]]:
    path = eval_run / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "true_x": float(row["true_x_m"]),
                    "true_y": float(row["true_y_m"]),
                    "pred_x": float(row["pred_x_m"]),
                    "pred_y": float(row["pred_y_m"]),
                    "error": float(row["error_m"]),
                }
            )
    return rows


def read_bs_positions(scene_dir: Path, bs_index: int | None = None) -> np.ndarray:
    label_path = scene_dir / "label" / "label_panel_0p5.json"
    if not label_path.exists():
        raise FileNotFoundError(label_path)
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    positions = np.asarray(payload["bs_positions"], dtype=np.float32)[:, :2]
    if bs_index is None:
        return positions
    if bs_index < 0 or bs_index >= len(positions):
        raise IndexError(f"BS index {bs_index} out of range for {scene_dir.name}: {len(positions)}")
    return positions[bs_index : bs_index + 1]


def read_floorplan_meta(scene_dir: Path) -> dict[str, object]:
    path = scene_dir / "floorplan" / "meta.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def meter_to_pixel(points_xy: np.ndarray, image_height: int, meta: dict[str, object]) -> np.ndarray:
    origin = np.asarray(meta.get("origin_xy_m", [0.0, 0.0]), dtype=np.float32)
    resolution = float(meta["resolution_m_per_pixel"])
    px = (points_xy[:, 0] - origin[0]) / resolution
    py = image_height - 1 - (points_xy[:, 1] - origin[1]) / resolution
    return np.column_stack([px, py])


def error_bin(error: float) -> ErrorBin:
    for item in ERROR_BINS:
        if item.contains(error):
            return item
    return ERROR_BINS[-1]


def bin_counts(errors: Iterable[float]) -> dict[str, int]:
    counts = {item.label: 0 for item in ERROR_BINS}
    for error in errors:
        counts[error_bin(float(error)).label] += 1
    return counts


def draw_scene(
    ax: Axes,
    scene: str,
    scene_root: Path,
    eval_run: Path,
    *,
    case_label: str | None = None,
    bs_index: int | None = None,
    use_label_floorplan: bool,
    show_legend: bool,
) -> dict[str, float | int]:
    scene_dir = scene_root / scene
    image_path = (
        scene_dir / "label_floorplan" / "label_panel_0p5.png"
        if use_label_floorplan
        else scene_dir / "floorplan" / "floorplan_1p60.png"
    )
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    meta = read_floorplan_meta(scene_dir)
    rows = read_predictions(eval_run)
    if not rows:
        raise ValueError(f"No predictions: {eval_run}")

    gt_m = np.asarray([[row["true_x"], row["true_y"]] for row in rows], dtype=np.float32)
    pred_m = np.asarray([[row["pred_x"], row["pred_y"]] for row in rows], dtype=np.float32)
    errors = np.asarray([row["error"] for row in rows], dtype=np.float32)
    gt_px = meter_to_pixel(gt_m, height, meta)
    pred_px = meter_to_pixel(pred_m, height, meta)
    bs_px = meter_to_pixel(read_bs_positions(scene_dir, bs_index), height, meta)

    ax.imshow(image, origin="upper")
    ax.set_xlim(-2, width + 2)
    ax.set_ylim(height + 2, -2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    for gt, pred, err in zip(gt_px, pred_px, errors):
        color = error_bin(float(err)).color
        arrow = FancyArrowPatch(
            posA=(gt[0], gt[1]),
            posB=(pred[0], pred[1]),
            arrowstyle="-|>",
            mutation_scale=7.0,
            linewidth=1.1,
            color=color,
            alpha=0.72,
            shrinkA=2.0,
            shrinkB=3.5,
            zorder=3,
        )
        ax.add_patch(arrow)

    ax.scatter(
        gt_px[:, 0],
        gt_px[:, 1],
        s=14,
        c="white",
        edgecolors="#3a3a3a",
        linewidths=0.45,
        zorder=5,
        label="UE GT",
    )
    for item in ERROR_BINS:
        mask = np.array([item.contains(float(value)) for value in errors])
        if np.any(mask):
            ax.scatter(
                pred_px[mask, 0],
                pred_px[mask, 1],
                s=36,
                facecolors=item.color,
                edgecolors="white",
                linewidths=0.7,
                alpha=0.96,
                zorder=6,
            )

    ax.scatter(
        bs_px[:, 0],
        bs_px[:, 1],
        marker="*",
        s=115,
        c="#f45b83",
        edgecolors="white",
        linewidths=0.8,
        zorder=7,
        label=f"BS x{len(bs_px)}",
    )

    counts = bin_counts(errors)
    mean_error = float(errors.mean())
    p90 = float(np.quantile(errors, 0.90))
    title_prefix = f"{case_label} | {scene}" if case_label else scene
    ax.set_title(
        f"{title_prefix} | n={len(errors)} | mean {mean_error:.2f}m p90 {p90:.2f}m",
        fontsize=9,
        pad=6,
    )

    if show_legend:
        handles: list[Line2D] = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=4.5,
                markerfacecolor="white",
                markeredgecolor="#3a3a3a",
                label="UE GT",
            )
        ]
        handles.extend(
            Line2D(
                [0],
                [0],
                color=item.color,
                marker="o",
                markersize=5,
                linewidth=1.2,
                label=f"{item.label}: {counts[item.label]}",
            )
            for item in ERROR_BINS
        )
        handles.append(
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="None",
                markersize=9,
                markerfacecolor="#f45b83",
                markeredgecolor="white",
                label=f"BS x{len(bs_px)}",
            )
        )
        legend = ax.legend(
            handles=handles,
            loc="upper right",
            fontsize=7,
            frameon=True,
            framealpha=0.92,
            title="Error bins",
            title_fontsize=8,
            borderpad=0.45,
            handlelength=1.5,
            labelspacing=0.35,
        )
        legend.get_frame().set_linewidth(0.4)

    return {
        "n": int(len(errors)),
        "mean_error_m": mean_error,
        "p90_error_m": p90,
        "max_error_m": float(errors.max()),
        "bs_count": int(len(bs_px)),
    }


def main() -> None:
    args = parse_args()
    cases = parse_cases(args)
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_dir = Path("runs/figures") / f"mpri_front3d_test_visuals-{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, float | int | str]] = {}
    for case in cases:
        fig, ax = plt.subplots(figsize=(6.2, 6.2))
        stats = draw_scene(
            ax,
            case.scene,
            args.scene_root,
            case.eval_run,
            case_label=case.label if args.case else None,
            bs_index=case.bs_index,
            use_label_floorplan=args.use_label_floorplan,
            show_legend=True,
        )
        fig.tight_layout(pad=0.5)
        out = output_dir / f"{case.slug}_mpri_test_predictions.png"
        fig.savefig(out, dpi=args.dpi)
        plt.close(fig)
        summary[case.slug] = {
            **stats,
            "figure": str(out),
            "eval_run": str(case.eval_run),
            "scene": case.scene,
            "label": case.label,
            "bs_index": "" if case.bs_index is None else int(case.bs_index),
        }

    overview_cols = args.overview_cols
    if overview_cols is None:
        overview_cols = 3 if args.case else len(cases)
    overview_cols = max(1, min(overview_cols, len(cases)))
    overview_rows = int(np.ceil(len(cases) / overview_cols))
    fig, axes = plt.subplots(
        overview_rows,
        overview_cols,
        figsize=(6.2 * overview_cols, 5.9 * overview_rows),
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for ax, case in zip(flat_axes, cases):
        draw_scene(
            ax,
            case.scene,
            args.scene_root,
            case.eval_run,
            case_label=case.label if args.case else None,
            bs_index=case.bs_index,
            use_label_floorplan=args.use_label_floorplan,
            show_legend=True,
        )
    for ax in flat_axes[len(cases) :]:
        ax.axis("off")
    fig.tight_layout(pad=0.4, w_pad=0.6)
    overview = output_dir / f"front3d_mpri_test_predictions_overview_{overview_rows}x{overview_cols}.png"
    fig.savefig(overview, dpi=args.dpi)
    plt.close(fig)
    summary["overview"] = {"figure": str(overview)}

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(output_dir)


if __name__ == "__main__":
    main()
