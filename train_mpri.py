from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
import yaml
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from src.data import CFRDataset, NormalizeTransform, split_indices_by_group
from src.metrics import METRIC_COLUMNS, localization_metrics
from src.model import MPRINet


DEFAULT_CONFIG: Dict[str, Any] = {
    "experiment": {
        "name": "mpri-baseline",
        "output_root": "runs",
    },
    "data": {
        "dataset": "data/csi/csi_strict.h5",
        "train_split": 0.7,
        "val_split": 0.2,
        "test_split": 0.1,
        "split_strategy": "position_level",
    },
    "model": {
        "dropout": 0.3,
        "output_dim": 2,
    },
    "optimizer": {
        "type": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
    },
    "scheduler": {
        "type": "StepLR",
        "step_size": 50,
        "gamma": 0.3,
    },
    "loss": {
        "type": "rmse",
        "eps": 1e-8,
    },
    "training": {
        "epochs": 400,
        "batch_size": 32,
        "seed": 3407,
        "device": "cuda",
        "amp": False,
        "max_norm": 5.0,
    },
    "logging": {
        "save_loss_figure": True,
        "loss_figure_name": "loss_curve.png",
        "evaluate_test_each_epoch": False,
    },
}


class TeeStream:
    """同时向终端和日志文件写入。"""

    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


@dataclass
class Metrics:
    """训练/评估指标"""
    loss: float      # RMSE Loss
    rmse: float      # Position RMSE (m)
    mean_err: float  # Mean Euclidean Error (m)
    mse_m2: float = float("nan")
    median_error_m: float = float("nan")
    cdf_1m: float = float("nan")
    cdf_2m: float = float("nan")
    p90_error_m: float = float("nan")
    p95_error_m: float = float("nan")
    p99_error_m: float = float("nan")
    valid_sample_count: int = 0


def metrics_from_errors(loss: float, error_parts: list[np.ndarray]) -> Metrics:
    if not error_parts:
        return Metrics(float("nan"), float("nan"), float("nan"))
    errors = np.concatenate(error_parts, axis=0)
    stats = localization_metrics(errors)
    return Metrics(
        loss=loss,
        rmse=float(stats["rmse_m"]),
        mean_err=float(stats["mean_error_m"]),
        mse_m2=float(stats["mse_m2"]),
        median_error_m=float(stats["median_error_m"]),
        cdf_1m=float(stats["cdf_1m"]),
        cdf_2m=float(stats["cdf_2m"]),
        p90_error_m=float(stats["p90_error_m"]),
        p95_error_m=float(stats["p95_error_m"]),
        p99_error_m=float(stats["p99_error_m"]),
        valid_sample_count=int(stats["valid_sample_count"]),
    )


def prefixed_metric_row(prefix: str, metrics: Metrics) -> dict[str, float | int]:
    return {
        f"{prefix}_loss": metrics.loss,
        f"{prefix}_rmse": metrics.rmse,
        f"{prefix}_mean_err": metrics.mean_err,
        f"{prefix}_mean_error_m": metrics.mean_err,
        f"{prefix}_mse_m2": metrics.mse_m2,
        f"{prefix}_median_error_m": metrics.median_error_m,
        f"{prefix}_rmse_m": metrics.rmse,
        f"{prefix}_cdf_1m": metrics.cdf_1m,
        f"{prefix}_cdf_2m": metrics.cdf_2m,
        f"{prefix}_p90_error_m": metrics.p90_error_m,
        f"{prefix}_p95_error_m": metrics.p95_error_m,
        f"{prefix}_p99_error_m": metrics.p99_error_m,
        f"{prefix}_valid_sample_count": metrics.valid_sample_count,
    }


def prefixed_metric_fieldnames(prefix: str) -> list[str]:
    return [
        f"{prefix}_loss",
        f"{prefix}_rmse",
        f"{prefix}_mean_err",
        *[f"{prefix}_{column}" for column in METRIC_COLUMNS],
    ]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    loss_eps: float,
    scaler: GradScaler | None = None,
) -> Metrics:
    """训练一个epoch，使用位置RMSE作为损失

    支持混合精度训练(AMP)以减少GPU内存使用。
    """
    model.train()
    running_loss = 0.0
    n_samples = 0
    error_parts: list[np.ndarray] = []

    use_amp = scaler is not None and device.type == "cuda"

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_size = inputs.size(0)

        optimizer.zero_grad()

        if use_amp:
            with autocast():
                outputs = model(inputs)
                diff = outputs - targets
                sq_norm = torch.sum(diff ** 2, dim=-1)
                mse = sq_norm.mean()
                rmse_loss = torch.sqrt(mse + loss_eps)

            scaler.scale(rmse_loss).backward()
            if grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            diff = outputs - targets
            sq_norm = torch.sum(diff ** 2, dim=-1)
            mse = sq_norm.mean()
            rmse_loss = torch.sqrt(mse + loss_eps)

            rmse_loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # 统计
        running_loss += rmse_loss.item() * batch_size
        error_parts.append(torch.linalg.norm(diff.detach(), dim=-1).cpu().numpy())
        n_samples += batch_size

    if n_samples == 0:
        return Metrics(float("nan"), float("nan"), float("nan"))
    epoch_loss = running_loss / n_samples
    return metrics_from_errors(epoch_loss, error_parts)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_eps: float,
) -> Metrics:
    """评估模型，返回 RMSE 和 Mean Error"""
    model.eval()
    running_loss = 0.0
    n_samples = 0
    error_parts: list[np.ndarray] = []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)

        diff = outputs - targets
        sq_norm = torch.sum(diff ** 2, dim=-1)
        mse = sq_norm.mean()
        rmse_loss = torch.sqrt(mse + loss_eps)

        running_loss += rmse_loss.item() * inputs.size(0)
        error_parts.append(torch.linalg.norm(diff, dim=-1).cpu().numpy())
        n_samples += inputs.size(0)

    if n_samples == 0:
        return Metrics(float("nan"), float("nan"), float("nan"))
    epoch_loss = running_loss / n_samples
    return metrics_from_errors(epoch_loss, error_parts)


def build_dataloaders(
    dataset_path: Path,
    batch_size: int,
    train_split: float,
    val_split: float,
    test_split: float,
    split_strategy: str,
    seed: int,
    overfit_n: int | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any], Dict[str, Any]]:
    """构建训练和验证 DataLoader。

    使用位置级切分协议：
    - 同一位置（相同 x, y, z）的所有样本只进入 train 或 val 其中之一
    - 避免"位置泄漏"导致验证结果偏乐观
    """
    # 固定 numpy 随机种子，确保位置切分可复现
    rng = np.random.RandomState(seed)

    base_dataset = CFRDataset(dataset_path, preload=True)
    dataset_size = len(base_dataset)

    # 按 group 切分，主口径为 global_mobile_id（UE/location 级）。
    group_ids = base_dataset.group_ids(split_strategy)
    splits = split_indices_by_group(group_ids, train_split, val_split, test_split, seed)

    train_indices_tensor = torch.tensor(splits["train"], dtype=torch.long)
    val_indices_tensor = torch.tensor(splits["val"], dtype=torch.long)
    test_indices_tensor = torch.tensor(splits["test"], dtype=torch.long)

    print(f"数据集总样本数: {dataset_size}", flush=True)
    print(f"split_strategy: {split_strategy}", flush=True)
    print(f"唯一 group 数: {len(np.unique(group_ids))}", flush=True)
    print(f"训练样本数: {len(train_indices_tensor)}", flush=True)
    print(f"验证样本数: {len(val_indices_tensor)}", flush=True)
    print(f"测试样本数: {len(test_indices_tensor)}", flush=True)

    # Overfit mode (用于调试)
    if overfit_n is not None:
        overfit_n = min(overfit_n, train_indices_tensor.numel())
        train_indices_tensor = train_indices_tensor[:overfit_n]
        val_indices_tensor = train_indices_tensor.clone()
        test_indices_tensor = train_indices_tensor.clone()
        print(f"[调试模式] 使用 {overfit_n} 个样本", flush=True)

    # 在训练集上计算归一化统计量
    features = base_dataset.features_tensor()[train_indices_tensor]
    mean = features.mean(dim=(0, 2, 3))
    std = features.std(dim=(0, 2, 3))
    norm = NormalizeTransform(mean, std)
    base_dataset.transform = norm

    # 打印输入 shape 信息（论文版要求）
    sample_shape = features[0].shape
    print(f"单样本输入 shape: ({sample_shape[0]}, {sample_shape[1]}, {sample_shape[2]})", flush=True)
    print(f"成批输入 shape: (B, {sample_shape[0]}, {sample_shape[1]}, {sample_shape[2]})", flush=True)

    train_subset = Subset(base_dataset, train_indices_tensor.tolist())
    val_subset = Subset(base_dataset, val_indices_tensor.tolist())
    test_subset = Subset(base_dataset, test_indices_tensor.tolist())

    # DataLoader 使用独立的 torch Generator，与 numpy seed 分开
    torch_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch_generator,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    stats = {"mean": mean.cpu(), "std": std.cpu(), "label_dim": int(base_dataset.labels_tensor().shape[1])}
    split_info = {
        "train_indices": train_indices_tensor.tolist(),
        "val_indices": val_indices_tensor.tolist(),
        "test_indices": test_indices_tensor.tolist(),
        "split_strategy": split_strategy,
        "group_ids": np.asarray(group_ids).astype(int).tolist(),
        "num_unique_groups": int(np.unique(group_ids).size),
        "seed": seed,
        "train_split": train_split,
        "val_split": val_split,
        "test_split": test_split,
    }
    return train_loader, val_loader, test_loader, stats, split_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 MPRI 网络（YAML 配置驱动）")
    parser.add_argument("--config", type=Path, required=True, help="YAML 配置文件路径")

    parser.add_argument("--experiment-name", type=str, default=None, help="覆盖 experiment.name")
    parser.add_argument("--output-root", type=Path, default=None, help="覆盖 experiment.output_root")
    parser.add_argument("--dataset", type=Path, default=None, help="覆盖 data.dataset")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 training.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖 training.batch_size")
    parser.add_argument("--learning-rate", type=float, default=None, help="覆盖 optimizer.learning_rate")
    parser.add_argument("--weight-decay", type=float, default=None, help="覆盖 optimizer.weight_decay")
    parser.add_argument("--dropout", type=float, default=None, help="覆盖 model.dropout")
    parser.add_argument("--output-dim", type=int, default=None, help="覆盖 model.output_dim")
    parser.add_argument("--train-split", type=float, default=None, help="覆盖 data.train_split")
    parser.add_argument("--val-split", type=float, default=None, help="覆盖 data.val_split")
    parser.add_argument("--test-split", type=float, default=None, help="覆盖 data.test_split")
    parser.add_argument("--split-strategy", type=str, default=None, help="覆盖 data.split_strategy")
    parser.add_argument("--seed", type=int, default=None, help="覆盖 training.seed")
    parser.add_argument("--device", type=str, default=None, help="覆盖 training.device")
    parser.add_argument("--max-norm", type=float, default=None, help="覆盖 training.max_norm")
    parser.add_argument("--scheduler-step-size", type=int, default=None, help="覆盖 scheduler.step_size")
    parser.add_argument("--scheduler-gamma", type=float, default=None, help="覆盖 scheduler.gamma")
    parser.add_argument("--loss-eps", type=float, default=None, help="覆盖 loss.eps")
    parser.add_argument("--loss-figure-name", type=str, default=None, help="覆盖 logging.loss_figure_name")

    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", help="覆盖 training.amp=true")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false", help="覆盖 training.amp=false")
    parser.set_defaults(amp=None)

    figure_group = parser.add_mutually_exclusive_group()
    figure_group.add_argument(
        "--save-loss-figure",
        dest="save_loss_figure",
        action="store_true",
        help="覆盖 logging.save_loss_figure=true",
    )
    figure_group.add_argument(
        "--no-save-loss-figure",
        dest="save_loss_figure",
        action="store_false",
        help="覆盖 logging.save_loss_figure=false",
    )
    parser.set_defaults(save_loss_figure=None)

    test_eval_group = parser.add_mutually_exclusive_group()
    test_eval_group.add_argument(
        "--evaluate-test-each-epoch",
        dest="evaluate_test_each_epoch",
        action="store_true",
        help="覆盖 logging.evaluate_test_each_epoch=true",
    )
    test_eval_group.add_argument(
        "--no-evaluate-test-each-epoch",
        dest="evaluate_test_each_epoch",
        action="store_false",
        help="覆盖 logging.evaluate_test_each_epoch=false",
    )
    parser.set_defaults(evaluate_test_each_epoch=None)

    return parser.parse_args()


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError("配置文件顶层必须是字典结构")
    return data


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.experiment_name is not None:
        config["experiment"]["name"] = args.experiment_name
    if args.output_root is not None:
        config["experiment"]["output_root"] = str(args.output_root)
    if args.dataset is not None:
        config["data"]["dataset"] = str(args.dataset)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["optimizer"]["learning_rate"] = args.learning_rate
    if args.weight_decay is not None:
        config["optimizer"]["weight_decay"] = args.weight_decay
    if args.dropout is not None:
        config["model"]["dropout"] = args.dropout
    if args.output_dim is not None:
        config["model"]["output_dim"] = args.output_dim
    if args.train_split is not None:
        config["data"]["train_split"] = args.train_split
    if args.val_split is not None:
        config["data"]["val_split"] = args.val_split
    if args.test_split is not None:
        config["data"]["test_split"] = args.test_split
    if args.split_strategy is not None:
        config["data"]["split_strategy"] = args.split_strategy
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.max_norm is not None:
        config["training"]["max_norm"] = args.max_norm
    if args.scheduler_step_size is not None:
        config["scheduler"]["step_size"] = args.scheduler_step_size
    if args.scheduler_gamma is not None:
        config["scheduler"]["gamma"] = args.scheduler_gamma
    if args.loss_eps is not None:
        config["loss"]["eps"] = args.loss_eps
    if args.amp is not None:
        config["training"]["amp"] = bool(args.amp)
    if args.save_loss_figure is not None:
        config["logging"]["save_loss_figure"] = bool(args.save_loss_figure)
    if args.loss_figure_name is not None:
        config["logging"]["loss_figure_name"] = args.loss_figure_name
    if args.evaluate_test_each_epoch is not None:
        config["logging"]["evaluate_test_each_epoch"] = bool(args.evaluate_test_each_epoch)


def validate_and_normalize_config(config: Dict[str, Any]) -> None:
    required_sections = [
        "experiment",
        "data",
        "model",
        "optimizer",
        "scheduler",
        "loss",
        "training",
        "logging",
    ]
    for section in required_sections:
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"配置缺少 section: {section}")

    exp_name = str(config["experiment"].get("name", "")).strip()
    if not exp_name:
        raise ValueError("experiment.name 不能为空")

    config["experiment"]["output_root"] = str(config["experiment"].get("output_root", "runs"))
    config["data"]["dataset"] = str(config["data"].get("dataset", "data/csi/csi_strict.h5"))
    config["data"]["split_strategy"] = str(config["data"].get("split_strategy", "position_level"))
    config["data"]["train_split"] = float(config["data"].get("train_split", 1.0 - float(config["data"].get("val_split", 0.2)) - float(config["data"].get("test_split", 0.0))))
    config["data"]["val_split"] = float(config["data"].get("val_split", 0.2))
    config["data"]["test_split"] = float(config["data"].get("test_split", 0.0))

    config["model"]["dropout"] = float(config["model"].get("dropout", 0.3))
    config["model"]["output_dim"] = int(config["model"].get("output_dim", 2))

    config["optimizer"]["type"] = str(config["optimizer"].get("type", "AdamW"))
    config["optimizer"]["learning_rate"] = float(config["optimizer"].get("learning_rate", 1e-3))
    config["optimizer"]["weight_decay"] = float(config["optimizer"].get("weight_decay", 1e-4))

    config["scheduler"]["type"] = str(config["scheduler"].get("type", "StepLR"))
    config["scheduler"]["step_size"] = int(config["scheduler"].get("step_size", 50))
    config["scheduler"]["gamma"] = float(config["scheduler"].get("gamma", 0.3))

    config["loss"]["type"] = str(config["loss"].get("type", "rmse"))
    config["loss"]["eps"] = float(config["loss"].get("eps", 1e-8))

    config["training"]["epochs"] = int(config["training"].get("epochs", 400))
    config["training"]["batch_size"] = int(config["training"].get("batch_size", 32))
    config["training"]["seed"] = int(config["training"].get("seed", 3407))
    config["training"]["device"] = str(config["training"].get("device", "cuda"))
    config["training"]["amp"] = bool(config["training"].get("amp", False))
    config["training"]["max_norm"] = float(config["training"].get("max_norm", 5.0))

    config["logging"]["save_loss_figure"] = bool(config["logging"].get("save_loss_figure", True))
    config["logging"]["loss_figure_name"] = str(config["logging"].get("loss_figure_name", "loss_curve.png"))
    config["logging"]["evaluate_test_each_epoch"] = bool(
        config["logging"].get("evaluate_test_each_epoch", False)
    )

    if config["training"]["epochs"] <= 0:
        raise ValueError("training.epochs 必须大于 0")
    if config["training"]["batch_size"] <= 0:
        raise ValueError("training.batch_size 必须大于 0")
    if config["optimizer"]["learning_rate"] <= 0:
        raise ValueError("optimizer.learning_rate 必须大于 0")
    if config["scheduler"]["step_size"] <= 0:
        raise ValueError("scheduler.step_size 必须大于 0")
    if config["loss"]["eps"] <= 0:
        raise ValueError("loss.eps 必须大于 0")

    val_split = config["data"]["val_split"]
    if not 0.0 <= val_split < 1.0:
        raise ValueError("data.val_split 必须在 [0, 1) 区间")
    train_split = config["data"]["train_split"]
    test_split = config["data"]["test_split"]
    if not 0.0 < train_split <= 1.0:
        raise ValueError("data.train_split 必须在 (0, 1] 区间")
    if not 0.0 <= test_split < 1.0:
        raise ValueError("data.test_split 必须在 [0, 1) 区间")
    if not np.isclose(train_split + val_split + test_split, 1.0):
        raise ValueError("data.train_split + data.val_split + data.test_split 必须等于 1")
    if config["model"]["output_dim"] <= 0:
        raise ValueError("model.output_dim 必须大于 0")

    optimizer_type = config["optimizer"]["type"].lower()
    if optimizer_type not in {"adam", "adamw"}:
        raise ValueError(f"暂不支持的优化器类型: {config['optimizer']['type']}")

    scheduler_type = config["scheduler"]["type"].lower()
    if scheduler_type != "steplr":
        raise ValueError(f"暂不支持的学习率调度器类型: {config['scheduler']['type']}")

    loss_type = config["loss"]["type"].lower()
    if loss_type != "rmse":
        raise ValueError(f"暂不支持的损失函数类型: {config['loss']['type']}")


def sanitize_experiment_name(name: str) -> str:
    cleaned = re.sub(r"\s+", "-", name.strip())
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "-", cleaned)
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return cleaned or "experiment"


def create_run_dir(config: Dict[str, Any]) -> Path:
    output_root = Path(config["experiment"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    exp_name = sanitize_experiment_name(config["experiment"]["name"])
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    base_name = f"{exp_name}-{timestamp}"

    run_dir = output_root / base_name
    index = 1
    while run_dir.exists():
        run_dir = output_root / f"{base_name}-{index:02d}"
        index += 1

    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    (run_dir / "tensorboard").mkdir(parents=True, exist_ok=False)
    return run_dir


def build_effective_config(
    config: Dict[str, Any],
    config_source: Path,
    run_dir: Path,
    device: torch.device,
    started_at: str,
) -> Dict[str, Any]:
    effective = deepcopy(config)
    effective["runtime"] = {
        "config_source": str(config_source),
        "run_dir": str(run_dir),
        "started_at": started_at,
        "device": str(device),
    }
    return effective


def build_checkpoint_payload(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: StepLR,
    stats: Dict[str, Any],
    split_info: Dict[str, Any],
    effective_config: Dict[str, Any],
    best_val_rmse: float,
) -> Dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "statistics": stats,
        "split": split_info,
        "config": effective_config,
        "best_val_rmse": best_val_rmse,
    }


def run_training(config: Dict[str, Any], config_source: Path, run_dir: Path) -> None:
    seed = int(config["training"]["seed"])
    dataset_path = Path(config["data"]["dataset"])
    epochs = int(config["training"]["epochs"])
    batch_size = int(config["training"]["batch_size"])
    train_split = float(config["data"]["train_split"])
    val_split = float(config["data"]["val_split"])
    test_split = float(config["data"]["test_split"])
    split_strategy = str(config["data"]["split_strategy"])
    dropout = float(config["model"]["dropout"])
    output_dim = int(config["model"]["output_dim"])
    lr = float(config["optimizer"]["learning_rate"])
    weight_decay = float(config["optimizer"]["weight_decay"])
    max_norm = float(config["training"]["max_norm"])
    loss_eps = float(config["loss"]["eps"])
    use_amp = bool(config["training"]["amp"])
    evaluate_test_each_epoch = bool(config["logging"].get("evaluate_test_each_epoch", False))

    checkpoints_dir = run_dir / "checkpoints"
    best_ckpt_path = checkpoints_dir / "best.pt"
    last_ckpt_path = checkpoints_dir / "last.pt"
    metrics_csv_path = run_dir / "metrics.csv"
    tensorboard_dir = run_dir / "tensorboard"
    config_snapshot_path = run_dir / "config.yaml"

    loss_figure_path: Path | None = None
    if bool(config["logging"]["save_loss_figure"]):
        loss_figure_path = run_dir / str(config["logging"]["loss_figure_name"])

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    requested_device = str(config["training"]["device"])
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")

    started_at = datetime.now().isoformat(timespec="seconds")
    effective_config = build_effective_config(config, config_source, run_dir, device, started_at)
    with config_snapshot_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(effective_config, handle, sort_keys=False, allow_unicode=True)

    print(f"实验目录: {run_dir}", flush=True)
    print(f"配置快照已保存: {config_snapshot_path}", flush=True)
    print(f"使用设备: {device}", flush=True)
    if use_amp and device.type == "cuda":
        print("使用混合精度训练(AMP)", flush=True)

    train_loader, val_loader, test_loader, stats, split_info = build_dataloaders(
        dataset_path,
        batch_size,
        train_split,
        val_split,
        test_split,
        split_strategy,
        seed,
    )

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    print(f"样本数: 训练 = {n_train}, 验证 = {n_val}, 测试 = {n_test}", flush=True)

    in_channels = int(stats["mean"].shape[0])
    print(f"输入通道数: {in_channels}", flush=True)
    print(f"输出维度: {output_dim}", flush=True)
    if int(stats["label_dim"]) != output_dim:
        raise ValueError(f"model.output_dim={output_dim} 与数据标签维度 {stats['label_dim']} 不一致")

    model = MPRINet(dropout=dropout, in_channels=in_channels, output_dim=output_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_params:,}", flush=True)

    optimizer_type = str(config["optimizer"]["type"]).lower()
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(
        optimizer,
        step_size=int(config["scheduler"]["step_size"]),
        gamma=float(config["scheduler"]["gamma"]),
    )

    scaler = GradScaler() if use_amp and device.type == "cuda" else None

    metrics_header = [
        "epoch",
        "lr",
        *prefixed_metric_fieldnames("train"),
        *prefixed_metric_fieldnames("val"),
        *prefixed_metric_fieldnames("test"),
        "epoch_time_s",
        "is_best",
    ]

    history = {"train_rmse": [], "val_rmse": [], "test_rmse": []}
    best_selection_rmse = float("inf")
    best_epoch = 0

    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    try:
        with metrics_csv_path.open("w", encoding="utf-8", newline="") as csv_handle:
            csv_writer = csv.DictWriter(csv_handle, fieldnames=metrics_header)
            csv_writer.writeheader()

            print("\n=== 开始训练 ===", flush=True)
            for epoch in range(1, epochs + 1):
                start = time.time()
                current_lr = float(optimizer.param_groups[0]["lr"])

                train_metrics = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    max_norm,
                    loss_eps,
                    scaler,
                )
                val_metrics = evaluate(model, val_loader, device, loss_eps)
                if evaluate_test_each_epoch and len(test_loader.dataset) > 0:
                    test_metrics = evaluate(model, test_loader, device, loss_eps)
                else:
                    test_metrics = Metrics(float("nan"), float("nan"), float("nan"))
                if len(val_loader.dataset) > 0:
                    selection_metrics = val_metrics
                elif evaluate_test_each_epoch and len(test_loader.dataset) > 0:
                    selection_metrics = test_metrics
                else:
                    selection_metrics = train_metrics

                duration = time.time() - start
                is_best = selection_metrics.rmse < best_selection_rmse
                if is_best:
                    best_selection_rmse = selection_metrics.rmse
                    best_epoch = epoch
                    torch.save(
                        build_checkpoint_payload(
                            epoch,
                            model,
                            optimizer,
                            scheduler,
                            stats,
                            split_info,
                            effective_config,
                            best_selection_rmse,
                        ),
                        best_ckpt_path,
                    )

                metric_row: dict[str, float | int] = {"epoch": epoch, "lr": current_lr}
                metric_row.update(prefixed_metric_row("train", train_metrics))
                metric_row.update(prefixed_metric_row("val", val_metrics))
                metric_row.update(prefixed_metric_row("test", test_metrics))
                metric_row.update({"epoch_time_s": duration, "is_best": int(is_best)})
                csv_writer.writerow(metric_row)
                csv_handle.flush()

                writer.add_scalar("train/loss", train_metrics.loss, epoch)
                writer.add_scalar("train/rmse", train_metrics.rmse, epoch)
                writer.add_scalar("train/mean_error", train_metrics.mean_err, epoch)
                if len(val_loader.dataset) > 0:
                    writer.add_scalar("val/loss", val_metrics.loss, epoch)
                    writer.add_scalar("val/rmse", val_metrics.rmse, epoch)
                    writer.add_scalar("val/mean_error", val_metrics.mean_err, epoch)
                if evaluate_test_each_epoch and len(test_loader.dataset) > 0:
                    writer.add_scalar("test/loss", test_metrics.loss, epoch)
                    writer.add_scalar("test/rmse", test_metrics.rmse, epoch)
                    writer.add_scalar("test/mean_error", test_metrics.mean_err, epoch)
                writer.add_scalar("train/lr", current_lr, epoch)

                message = (
                    f"[Epoch {epoch:03d}/{epochs:03d}] "
                    f"train_loss: {train_metrics.loss:.4f} | "
                    f"val_loss: {val_metrics.loss:.4f} | "
                    f"train_rmse: {train_metrics.rmse:.3f}m | "
                    f"val_rmse: {val_metrics.rmse:.3f}m | "
                    f"val_mean_err: {val_metrics.mean_err:.3f}m"
                )
                if evaluate_test_each_epoch and len(test_loader.dataset) > 0:
                    message += (
                        f" | test_rmse: {test_metrics.rmse:.3f}m"
                        f" | test_mean_err: {test_metrics.mean_err:.3f}m"
                    )
                message += f" | lr: {current_lr:.6g} | time: {duration:.1f}s"
                print(message, flush=True)

                history["train_rmse"].append(train_metrics.rmse)
                history["val_rmse"].append(val_metrics.rmse)
                history["test_rmse"].append(test_metrics.rmse)

                scheduler.step()

        final_epoch = epochs
        torch.save(
            build_checkpoint_payload(
                final_epoch,
                model,
                optimizer,
                scheduler,
                stats,
                split_info,
                effective_config,
                    best_selection_rmse,
                ),
            last_ckpt_path,
        )
    finally:
        writer.flush()
        writer.close()

    print("\n=== 训练完成 ===", flush=True)
    if len(val_loader.dataset) > 0:
        selection_name = "验证"
    elif evaluate_test_each_epoch and len(test_loader.dataset) > 0:
        selection_name = "测试"
    else:
        selection_name = "训练"
    print(f"最佳{selection_name} RMSE: {best_selection_rmse:.3f}m (Epoch {best_epoch})", flush=True)
    print(f"最佳模型: {best_ckpt_path}", flush=True)
    print(f"最终模型: {last_ckpt_path}", flush=True)
    print(f"指标文件: {metrics_csv_path}", flush=True)
    print(f"TensorBoard: {tensorboard_dir}", flush=True)

    if loss_figure_path is not None:
        loss_figure_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            epochs_axis = range(1, len(history["train_rmse"]) + 1)
            plt.figure(figsize=(10, 6))
            plt.plot(epochs_axis, history["train_rmse"], label="Train RMSE", linewidth=1.5)
            plt.plot(epochs_axis, history["val_rmse"], label="Val RMSE", linewidth=1.5)
            if evaluate_test_each_epoch and any(np.isfinite(history["test_rmse"])):
                plt.plot(epochs_axis, history["test_rmse"], label="Test RMSE", linewidth=1.5)
            plt.xlabel("Epoch")
            plt.ylabel("Position RMSE (m)")
            plt.title("MPRI Training Curve")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.legend()
            plt.tight_layout()
            plt.savefig(loss_figure_path, dpi=200)
            plt.close()
            print(f"损失曲线已保存: {loss_figure_path}", flush=True)
        except Exception as exc:
            print(f"绘制损失曲线失败: {exc}", flush=True)


def main() -> None:
    args = parse_args()

    yaml_config = load_yaml_config(args.config)
    config = deep_update(DEFAULT_CONFIG, yaml_config)
    apply_cli_overrides(config, args)
    validate_and_normalize_config(config)

    run_dir = create_run_dir(config)
    train_log_path = run_dir / "train.log"

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with train_log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            run_training(config, args.config, run_dir)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
