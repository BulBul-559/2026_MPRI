#!/usr/bin/env python3
"""Sionna CFR → MPRI CSI 数据适配器。

将 Sionna 输出的原始 CFR 复数向量转换为 MPRI 训练/评估脚本可直接读取的
HDF5 数据格式，输出幅度-相位交织的 CSI 特征。

输入格式 (Sionna):
    channel/cfr: (num_aps, num_rx, num_tx_ant, num_rx_ant, num_sc) complex64
    channel/has_signal: (num_aps, num_rx) bool
    topology/rx_positions: (num_rx, 3) float32

输出格式 (MPRI):
    features: (W, H, C, N) float32
        W = num_ant_pairs * 2  (幅度-相位交织)
        H = num_subcarriers
        C = num_aps
        N = num_samples
    labels/position: (N, 3) float32

输出两个文件:
    1. csi_strict.h5 — 仅保留所有 AP 都有信号的位置
    2. csi_padded.h5 — 保留至少一个 AP 有信号的位置，缺失 AP 零填充
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 Sionna CFR 转换为 MPRI 可用的 CSI HDF5 数据集")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw"),
        help="原始 Sionna HDF5 文件或目录（默认: data/raw）",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.h5",
        help="当 --input 是目录时，匹配输入文件的 glob 模式（默认: *.h5）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/csi"),
        help="输出目录（默认: data/csi）",
    )
    parser.add_argument(
        "--strict-name",
        type=str,
        default="csi_strict.h5",
        help="严格策略输出文件名（默认: csi_strict.h5）",
    )
    parser.add_argument(
        "--padded-name",
        type=str,
        default="csi_padded.h5",
        help="零填充策略输出文件名（默认: csi_padded.h5）",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="转换时每个 AP 的 RX 分块大小（默认: 512）",
    )
    parser.add_argument(
        "--compression",
        choices=["none", "lzf", "gzip"],
        default="lzf",
        help="输出 HDF5 压缩方式（默认: lzf）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若输出文件已存在则覆盖",
    )
    return parser.parse_args()


def resolve_input_file(input_path: Path, pattern: str) -> Path:
    """将输入路径解析为单个 HDF5 文件。"""
    if input_path.is_file():
        return input_path

    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")
    if not input_path.is_dir():
        raise ValueError(f"输入路径不是文件或目录: {input_path}")

    matched = sorted(p for p in input_path.glob(pattern) if p.is_file())
    if len(matched) == 0:
        raise FileNotFoundError(f"目录 {input_path} 下未找到匹配 {pattern} 的文件")
    if len(matched) > 1:
        candidates = "\n".join(str(p) for p in matched)
        raise ValueError(
            "输入目录匹配到多个文件，请通过 --input 指定具体文件，或缩小 --glob:\n"
            f"{candidates}"
        )

    return matched[0]


def load_raw_data(handle: h5py.File) -> Dict[str, Any]:
    """从打开的 HDF5 文件加载原始 CFR 元信息。"""
    required_keys = ["channel/cfr", "channel/has_signal", "topology/rx_positions"]
    for key in required_keys:
        if key not in handle:
            raise KeyError(f"缺少必需数据集: {key}")

    cfr = handle["channel/cfr"]  # h5py dataset, lazy read
    has_signal = np.asarray(handle["channel/has_signal"], dtype=bool)
    rx_positions = np.asarray(handle["topology/rx_positions"], dtype=np.float32)

    return {
        "cfr": cfr,
        "has_signal": has_signal,
        "rx_positions": rx_positions,
    }


def validate_raw_data(data: Dict[str, Any]) -> None:
    """检查输入数据形状是否符合预期。"""
    cfr_shape = data["cfr"].shape
    has_signal_shape = data["has_signal"].shape
    rx_positions_shape = data["rx_positions"].shape

    if len(cfr_shape) != 5:
        raise ValueError(
            "channel/cfr 维度错误，期望 (num_aps, num_rx, num_tx_ant, num_rx_ant, num_sc)，"
            f"实际为 {cfr_shape}"
        )

    num_aps, num_rx, _, _, _ = cfr_shape
    if has_signal_shape != (num_aps, num_rx):
        raise ValueError(
            "channel/has_signal 维度错误，"
            f"期望 {(num_aps, num_rx)}，实际为 {has_signal_shape}"
        )

    if len(rx_positions_shape) != 2 or rx_positions_shape[0] != num_rx or rx_positions_shape[1] < 3:
        raise ValueError(
            "topology/rx_positions 维度错误，"
            f"期望 ({num_rx}, >=3)，实际为 {rx_positions_shape}"
        )


def cfr_to_mp_interleaved(cfr: np.ndarray) -> np.ndarray:
    """将复数 CFR 转换为幅度-相位交织格式

    Args:
        cfr: (num_rx, num_tx_ant, num_rx_ant, num_sc) complex64

    Returns:
        csi: (num_rx, num_sc, num_ant_pairs * 2) float32
             沿最后一维: [amp_1, phase_1, amp_2, phase_2, ...]
    """
    num_rx, num_tx_ant, num_rx_ant, num_sc = cfr.shape
    num_ant_pairs = num_tx_ant * num_rx_ant

    # 展平天线维度: (num_rx, num_ant_pairs, num_sc)
    cfr_flat = cfr.reshape(num_rx, num_ant_pairs, num_sc)

    # 计算幅度和相位
    amplitude = np.abs(cfr_flat)  # >= 0
    phase = np.angle(cfr_flat)  # [-π, π]

    # 交织: [amp_1, phase_1, amp_2, phase_2, ...]
    csi = np.zeros((num_rx, num_ant_pairs * 2, num_sc), dtype=np.float32)
    csi[:, 0::2, :] = amplitude  # 偶数索引: 幅度
    csi[:, 1::2, :] = phase  # 奇数索引: 相位

    # 转置为 (num_rx, num_sc, num_ant_pairs * 2) = (N, H, W)
    csi = csi.transpose(0, 2, 1)
    return csi


def build_features(
    data: Dict[str, Any],
    rx_mask: np.ndarray,
    zero_fill: bool = False,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """构建多 AP 合并的 CSI 特征和标签

    Args:
        data: 原始数据字典
        rx_mask: 保留的 RX 位置掩码 (num_rx,)
        zero_fill: True 时对缺失信号的 AP 填充零；False 时跳过该位置
        chunk_size: 每次转换的 RX 分块大小

    Returns:
        features: (W, H, C, N) float32 — Fortran order
        labels: (N, 3) float32
    """
    cfr = data["cfr"]
    has_signal = data["has_signal"]
    rx_positions = data["rx_positions"]
    num_aps, _, num_tx_ant, num_rx_ant, num_sc = cfr.shape
    num_ant_pairs = num_tx_ant * num_rx_ant
    width = num_ant_pairs * 2

    rx_indices = np.where(rx_mask)[0]
    num_samples = int(rx_indices.shape[0])
    positions = rx_positions[rx_indices, :3].astype(np.float32)

    # 输出 shape: (W, H, C, N)
    features = np.zeros((width, num_sc, num_aps, num_samples), dtype=np.float32)

    if num_samples == 0:
        return features, positions

    if chunk_size <= 0:
        raise ValueError(f"chunk_size 必须大于 0，当前值: {chunk_size}")

    for ap_idx in range(num_aps):
        if zero_fill:
            local_indices = np.where(has_signal[ap_idx, rx_indices])[0]
        else:
            local_indices = np.arange(num_samples, dtype=np.int64)

        if local_indices.size == 0:
            continue

        for start in range(0, local_indices.size, chunk_size):
            local_chunk = local_indices[start : start + chunk_size]
            rx_chunk = rx_indices[local_chunk]
            cfr_chunk = np.asarray(cfr[ap_idx, rx_chunk], dtype=np.complex64)
            csi_chunk = cfr_to_mp_interleaved(cfr_chunk)  # (chunk, H, W)

            # (chunk, H, W) -> (W, H, chunk)
            features[:, :, ap_idx, local_chunk] = csi_chunk.transpose(2, 1, 0)

    return features, positions


def save_output(
    output_path: Path,
    features: np.ndarray,
    labels: np.ndarray,
    config: Dict[str, Any] | None = None,
    compression: str = "lzf",
) -> None:
    """保存为 MPRI 兼容的 HDF5 格式"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compression_arg = None if compression == "none" else compression

    with h5py.File(output_path, "w") as f:
        create_kwargs: Dict[str, Any] = {"dtype": np.float32}
        if compression_arg is not None:
            create_kwargs["compression"] = compression_arg
            if compression_arg == "gzip":
                create_kwargs["compression_opts"] = 4

        f.create_dataset("features", data=features, **create_kwargs)
        labels_group = f.create_group("labels")
        labels_group.create_dataset("position", data=labels, **create_kwargs)
        if config:
            config_group = f.create_group("config")
            for k, v in config.items():
                config_group.attrs[k] = v


def main():
    args = parse_args()

    try:
        input_file = resolve_input_file(args.input, args.glob)
    except Exception as exc:
        print(f"错误: {exc}")
        sys.exit(1)

    strict_path = args.output_dir / args.strict_name
    padded_path = args.output_dir / args.padded_name
    for output_path in (strict_path, padded_path):
        if output_path.exists() and not args.overwrite:
            print(f"错误: 输出文件已存在: {output_path} (可加 --overwrite 覆盖)")
            sys.exit(1)

    print(f"加载原始数据: {input_file}")

    with h5py.File(input_file, "r") as handle:
        data = load_raw_data(handle)
        validate_raw_data(data)

        cfr = data["cfr"]
        has_signal = data["has_signal"]
        num_aps, num_rx, num_tx_ant, num_rx_ant, num_sc = cfr.shape
        num_ant_pairs = num_tx_ant * num_rx_ant

        print(f"  CFR shape: ({num_aps}, {num_rx}, {num_tx_ant}, {num_rx_ant}, {num_sc})")
        print(f"  天线对数: {num_ant_pairs} -> W={num_ant_pairs * 2} (幅度-相位交织)")
        print(f"  子载波数(Ns): {num_sc}")
        print(f"  AP 数(Ng): {num_aps} -> C={num_aps}")
        print()

        # 统计各 AP 信号覆盖率
        for ap_idx in range(num_aps):
            valid = int(has_signal[ap_idx].sum())
            if valid > 0:
                first_valid_idx = int(np.where(has_signal[ap_idx])[0][0])
                sample_mag = float(np.abs(cfr[ap_idx, first_valid_idx]).mean())
                print(f"  AP{ap_idx}: {valid}/{num_rx} 有效, 示例幅度均值={sample_mag:.6f}")
            else:
                print(f"  AP{ap_idx}: 0/{num_rx} 有效")

        strict_mask = has_signal.all(axis=0)  # 所有 AP 都有信号
        padded_mask = has_signal.any(axis=0)  # 至少一个 AP 有信号
        print(f"\n  所有 AP 都有信号: {int(strict_mask.sum())}")
        print(f"  至少一个 AP 有信号: {int(padded_mask.sum())}")
        print()

        print("=" * 50)
        print(f"生成文件 1: {strict_path.name} (严格策略)")
        features_strict, labels_strict = build_features(
            data,
            strict_mask,
            zero_fill=False,
            chunk_size=args.chunk_size,
        )
        strict_samples = int(labels_strict.shape[0])
        strict_config = {
            "num_aps": int(features_strict.shape[2]),
            "num_samples": strict_samples,
            "num_subcarriers": int(features_strict.shape[1]),
            "num_ant_pairs": int(num_ant_pairs),
            "format": "mp_interleaved",
            "mode": "strict",
            "source_file": str(input_file),
        }
        save_output(
            strict_path,
            features_strict,
            labels_strict,
            config=strict_config,
            compression=args.compression,
        )
        print(
            "  features: "
            f"(W={features_strict.shape[0]}, H={features_strict.shape[1]}, "
            f"C={features_strict.shape[2]}, N={strict_samples})"
        )
        print(f"  labels: ({strict_samples}, 3)")

        print()
        print("=" * 50)
        print(f"生成文件 2: {padded_path.name} (零填充策略)")
        features_padded, labels_padded = build_features(
            data,
            padded_mask,
            zero_fill=True,
            chunk_size=args.chunk_size,
        )
        padded_samples = int(labels_padded.shape[0])

        rx_indices = np.where(padded_mask)[0]
        for ap_idx in range(num_aps):
            filled = int((~has_signal[ap_idx, rx_indices]).sum())
            print(f"  AP{ap_idx}: 零填充 {filled}/{padded_samples} 个位置")

        padded_config = {
            "num_aps": int(features_padded.shape[2]),
            "num_samples": padded_samples,
            "num_subcarriers": int(features_padded.shape[1]),
            "num_ant_pairs": int(num_ant_pairs),
            "format": "mp_interleaved",
            "mode": "padded",
            "source_file": str(input_file),
        }
        save_output(
            padded_path,
            features_padded,
            labels_padded,
            config=padded_config,
            compression=args.compression,
        )
        print(
            "  features: "
            f"(W={features_padded.shape[0]}, H={features_padded.shape[1]}, "
            f"C={features_padded.shape[2]}, N={padded_samples})"
        )
        print(f"  labels: ({padded_samples}, 3)")

    print("\n转换完成!")
    print(f"严格策略输出: {strict_path}")
    print(f"零填充策略输出: {padded_path}")


if __name__ == "__main__":
    main()
