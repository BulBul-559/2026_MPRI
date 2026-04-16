from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler

from src.data import CFRDataset, NormalizeTransform
from src.model import MPRINet


@dataclass
class Metrics:
    """训练/评估指标"""
    loss: float      # RMSE Loss
    rmse: float      # Position RMSE (m)
    mean_err: float  # Mean Euclidean Error (m)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    scaler: GradScaler | None = None,
) -> Metrics:
    """训练一个epoch，使用位置RMSE作为损失

    支持混合精度训练(AMP)以减少GPU内存使用。
    """
    model.train()
    running_loss = 0.0
    running_sqerr = 0.0
    running_mean_err = 0.0
    n_samples = 0

    eps = 1e-8  # 防止 sqrt(0) 梯度问题
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
                rmse_loss = torch.sqrt(mse + eps)

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
            rmse_loss = torch.sqrt(mse + eps)

            rmse_loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # 统计
        running_loss += rmse_loss.item() * batch_size
        running_sqerr += sq_norm.sum().item()
        running_mean_err += torch.linalg.norm(diff, dim=-1).sum().item()
        n_samples += batch_size

    epoch_loss = running_loss / n_samples
    rmse = math.sqrt(running_sqerr / n_samples)
    mean_err = running_mean_err / n_samples
    return Metrics(epoch_loss, rmse, mean_err)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Metrics:
    """评估模型，返回 RMSE 和 Mean Error"""
    model.eval()
    running_loss = 0.0
    running_sqerr = 0.0
    running_mean_err = 0.0
    n_samples = 0
    eps = 1e-8

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)

        diff = outputs - targets
        sq_norm = torch.sum(diff ** 2, dim=-1)
        mse = sq_norm.mean()
        rmse_loss = torch.sqrt(mse + eps)

        running_loss += rmse_loss.item() * inputs.size(0)
        running_sqerr += sq_norm.sum().item()
        running_mean_err += torch.linalg.norm(diff, dim=-1).sum().item()
        n_samples += inputs.size(0)

    epoch_loss = running_loss / n_samples
    rmse = math.sqrt(running_sqerr / n_samples)
    mean_err = running_mean_err / n_samples
    return Metrics(epoch_loss, rmse, mean_err)


def build_dataloaders(
    dataset_path: Path,
    batch_size: int,
    val_split: float,
    seed: int,
    overfit_n: int | None = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, torch.Tensor], Dict[str, List[int]]]:
    """构建训练和验证 DataLoader。

    使用位置级切分协议：
    - 同一位置（相同 x, y, z）的所有样本只进入 train 或 val 其中之一
    - 避免"位置泄漏"导致验证结果偏乐观
    """
    # 固定 numpy 随机种子，确保位置切分可复现
    rng = np.random.RandomState(seed)

    base_dataset = CFRDataset(dataset_path, preload=True)
    dataset_size = len(base_dataset)

    # 获取位置信息和位置-样本映射
    unique_pos_ids, pos_to_indices = base_dataset.unique_positions()
    num_unique_positions = len(unique_pos_ids)

    # 按位置级切分：将位置（而非样本）分配到 train 或 val
    # 使用受 seed 控制的 rng 确保切分可复现
    shuffled_pos_ids = rng.permutation(unique_pos_ids)
    num_val_positions = max(1, int(num_unique_positions * val_split))
    num_train_positions = num_unique_positions - num_val_positions

    train_positions: Set[int] = set(shuffled_pos_ids[:num_train_positions])
    val_positions: Set[int] = set(shuffled_pos_ids[num_train_positions:])

    # 根据位置分配样本索引
    train_indices: List[int] = []
    val_indices: List[int] = []

    for pos_id in train_positions:
        train_indices.extend(pos_to_indices[pos_id])
    for pos_id in val_positions:
        val_indices.extend(pos_to_indices[pos_id])

    train_indices_tensor = torch.tensor(train_indices, dtype=torch.long)
    val_indices_tensor = torch.tensor(val_indices, dtype=torch.long)

    print(f"数据集总样本数: {dataset_size}", flush=True)
    print(f"唯一位置数: {num_unique_positions}", flush=True)
    print(f"训练位置数: {num_train_positions}, 训练样本数: {len(train_indices)}", flush=True)
    print(f"验证位置数: {num_val_positions}, 验证样本数: {len(val_indices)}", flush=True)

    # Overfit mode (用于调试)
    if overfit_n is not None:
        overfit_n = min(overfit_n, train_indices_tensor.numel())
        train_indices_tensor = train_indices_tensor[:overfit_n]
        val_indices_tensor = train_indices_tensor.clone()
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

    stats = {"mean": mean.cpu(), "std": std.cpu()}
    split_info = {
        "train_indices": train_indices_tensor.tolist(),
        "val_indices": val_indices_tensor.tolist(),
        "train_positions": list(train_positions),
        "val_positions": list(val_positions),
        "num_unique_positions": num_unique_positions,
        "seed": seed,
        "val_split": val_split,
    }
    return train_loader, val_loader, stats, split_info


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 MPRI 网络（论文版）")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/csi/csi_strict.h5"),
        help="HDF5 数据集文件路径（推荐 .h5）",
    )
    parser.add_argument("--epochs", type=int, default=400, help="训练轮数（论文默认400）")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("checkpoints/mpri_paper_baseline.pt"),
        help="模型保存路径",
    )
    parser.add_argument("--max-norm", type=float, default=5.0, help="梯度裁剪阈值")
    parser.add_argument(
        "--loss-figure",
        type=str,
        default="plots/loss_curve_paper_baseline.png",
        help="损失曲线保存路径",
    )
    parser.add_argument("--amp", action="store_true", help="使用混合精度训练减少GPU内存")
    args = parser.parse_args()

    # 固定随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}", flush=True)

    # 混合精度训练
    scaler = GradScaler() if args.amp and device.type == "cuda" else None
    if scaler:
        print("使用混合精度训练(AMP)", flush=True)

    # 构建数据加载器
    train_loader, val_loader, stats, split_info = build_dataloaders(
        args.dataset, args.batch_size, args.val_split, args.seed
    )

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    print(f"样本数: 训练 = {n_train}, 验证 = {n_val}", flush=True)

    # 输入通道数由数据自动推断
    in_channels = int(stats["mean"].shape[0])
    print(f"输入通道数: {in_channels}", flush=True)

    # 构建模型
    model = MPRINet(dropout=args.dropout, in_channels=in_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_params:,}", flush=True)

    # 优化器: AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # 学习率调度: StepLR (step_size=50, gamma=0.3) 与论文一致
    scheduler = StepLR(optimizer, step_size=50, gamma=0.3)

    # 训练循环
    best_val_rmse = float("inf")
    save_path = args.save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 简化 history：loss 和 rmse 是同一个值（使用 RMSE loss），只记录 rmse
    history = {"train_rmse": [], "val_rmse": []}

    print("\n=== 开始训练 ===", flush=True)
    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args.max_norm, scaler)
        val_metrics = evaluate(model, val_loader, device)

        scheduler.step()
        duration = time.time() - start

        # 日志输出
        print(
            f"[Epoch {epoch:03d}] "
            f"Train RMSE: {train_metrics.rmse:.3f}m | "
            f"Val RMSE: {val_metrics.rmse:.3f}m | "
            f"Val MeanErr: {val_metrics.mean_err:.3f}m | "
            f"{duration:.1f}s",
            flush=True,
        )

        # 记录历史（loss = rmse）
        history["train_rmse"].append(train_metrics.rmse)
        history["val_rmse"].append(val_metrics.rmse)

        # 保存最佳模型（按 Val RMSE）
        if val_metrics.rmse < best_val_rmse:
            best_val_rmse = val_metrics.rmse
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "statistics": stats,
                    "split": split_info,
                    "config": vars(args),
                    "best_val_rmse": best_val_rmse,
                },
                save_path,
            )

    print(f"\n=== 训练完成 ===", flush=True)
    print(f"最佳验证 RMSE: {best_val_rmse:.3f}m", flush=True)
    print(f"模型已保存至: {save_path}", flush=True)

    # 绘制损失曲线
    figure_target = (args.loss_figure or "").strip()
    if figure_target:
        figure_path = Path(figure_target)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            epochs_axis = range(1, len(history["train_rmse"]) + 1)
            plt.figure(figsize=(10, 6))
            plt.plot(epochs_axis, history["train_rmse"], label="Train RMSE", linewidth=1.5)
            plt.plot(epochs_axis, history["val_rmse"], label="Val RMSE", linewidth=1.5)
            plt.xlabel("Epoch")
            plt.ylabel("Position RMSE (m)")
            plt.title("MPRI Paper Baseline Training Curve")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.legend()
            plt.tight_layout()
            plt.savefig(figure_path, dpi=200)
            plt.close()
            print(f"损失曲线已保存至 {figure_path}", flush=True)
        except Exception as exc:
            print(f"绘制损失曲线失败: {exc}", flush=True)


if __name__ == "__main__":
    main()