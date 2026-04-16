from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data import CFRDataset, NormalizeTransform
from src.model import MPRINet


import os
import pathlib
from pathlib import Path

import torch

def load_model(checkpoint_path: Path, device: torch.device) -> tuple[MPRINet, dict]:
    """
    加载模型 checkpoint，兼容 Linux 保存、Windows 加载时的 PosixPath 问题。
    """
    # 兼容：在 Windows 上加载 Linux 保存的 checkpoint（里面有 PosixPath）
    if os.name == "nt":  # 'nt' 代表 Windows
        _orig_posixpath = pathlib.PosixPath
        try:
            # 让 pickle 里记录的 PosixPath 反序列化成 WindowsPath
            pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[attr-defined]
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        finally:
            # 恢复原来的 PosixPath，避免影响其他代码
            pathlib.PosixPath = _orig_posixpath
    else:
        # Linux/macOS 正常加载
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    stats = checkpoint.get("statistics")
    in_channels = int(stats["mean"].shape[0])
    config = checkpoint.get("config", {})

    model = MPRINet(
        dropout=config.get("dropout", 0.3),
        in_channels=in_channels,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model, checkpoint

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    criterion = nn.MSELoss()

    losses = []
    preds = []
    targets = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        losses.append(loss.detach().cpu())
        preds.append(outputs.detach().cpu())
        targets.append(labels.detach().cpu())

    predictions = torch.cat(preds, dim=0)
    ground_truth = torch.cat(targets, dim=0)
    loss = torch.stack(losses).mean().item()

    diff = predictions - ground_truth
    per_sample_error = torch.linalg.norm(diff, dim=-1)
    rmse = torch.sqrt(torch.mean(torch.sum(diff**2, dim=-1))).item()
    mean_err = per_sample_error.mean().item()
    median_err = per_sample_error.median().item()
    p90_err = torch.quantile(per_sample_error, 0.9).item()

    axis_mae = torch.mean(torch.abs(diff), dim=0).tolist()

    return {
        "loss": loss,
        "rmse": rmse,
        "mean_error": mean_err,
        "median_error": median_err,
        "p90_error": p90_err,
        "axis_mae": axis_mae,
        "predictions": predictions.numpy(),
        "ground_truth": ground_truth.numpy(),
        "per_sample_error": per_sample_error.numpy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained MPRI model.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/mpri_model.pt"),
        help="Path to the saved checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/csi/csi_strict.h5"),
        help="Path to the HDF5 dataset file (prefer .h5).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "all"],
        default="val",
        help="Which split to evaluate. Uses indices stored in the checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--save-errors",
        type=Path,
        default=None,
        help="Optional path to save per-sample predictions and errors as .csv.",
    )
    parser.add_argument(
        "--save-preds",
        type=Path,
        default=Path("results/predictions.csv"),
        help="Path to save predictions with ground truth for the selected split (CSV).",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.checkpoint, device)

    stats = checkpoint.get("statistics")
    if stats is None:
        raise ValueError("Checkpoint does not contain normalisation statistics.")

    dataset = CFRDataset(args.dataset, preload=True)
    dataset.transform = NormalizeTransform(
        torch.as_tensor(stats["mean"]), torch.as_tensor(stats["std"])
    )

    split_info = checkpoint.get("split")
    if args.split == "all" or split_info is None:
        indices = list(range(len(dataset)))
    else:
        key = f"{args.split}_indices"
        if key not in split_info:
            raise ValueError(f"Split '{args.split}' not found in checkpoint.")
        indices = split_info[key]

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False)

    results = evaluate_model(model, loader, device)

    print("=== Evaluation Results (meters) ===")
    print(f"Samples evaluated : {len(indices)}")
    print(f"Loss (MSE)        : {results['loss']:.4f}")
    print(f"RMSE              : {results['rmse']:.3f}")
    print(f"Mean Error        : {results['mean_error']:.3f}")
    print(f"Median Error      : {results['median_error']:.3f}")
    print(f"90th Percentile   : {results['p90_error']:.3f}")
    print(
        "Axis MAE (|dx|, |dy|, |dz|): "
        f"{results['axis_mae'][0]:.3f}, {results['axis_mae'][1]:.3f}, {results['axis_mae'][2]:.3f} m"
    )

    errors = results["per_sample_error"]

    print("\n=== Error Statistics ===")
    mean_err = float(np.mean(errors))
    median_err = float(np.median(errors))
    p90_err = float(np.quantile(errors, 0.9))
    print(
        f"Mean Error: {mean_err:.3f} m, "
        f"Median Error: {median_err:.3f} m, "
        f"P90 Error: {p90_err:.3f} m"
    )

    if args.save_errors:
        args.save_errors.parent.mkdir(parents=True, exist_ok=True)
        data = np.hstack(
            [
                results["ground_truth"],
                results["predictions"],
                results["per_sample_error"].reshape(-1, 1),
            ]
        )
        header = "x_true,y_true,z_true,x_pred,y_pred,z_pred,error_m"
        np.savetxt(args.save_errors, data, delimiter=",", header=header, comments="")
        print(f"详细误差已保存至 {args.save_errors}")

    if args.save_preds:
        args.save_preds.parent.mkdir(parents=True, exist_ok=True)
        idx_col = np.asarray(indices, dtype=np.int64).reshape(-1, 1)
        combined = np.hstack(
            [
                idx_col,
                results["ground_truth"],
                results["predictions"],
                results["per_sample_error"].reshape(-1, 1),
            ]
        )
        header = "index,x_true,y_true,z_true,x_pred,y_pred,z_pred,error_m"
        np.savetxt(args.save_preds, combined, delimiter=",", header=header, comments="")
        print(f"预测与真值已保存至 {args.save_preds}")


if __name__ == "__main__":
    main()
