from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetStatistics:
    """数据集归一化统计量"""
    mean: torch.Tensor
    std: torch.Tensor


class NormalizeTransform:
    """通道级归一化变换"""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / (self.std + 1e-6)


class CFRDataset(Dataset):
    """从 HDF5 文件加载 CSI/CFR 数据的数据集"""

    def __init__(
        self,
        data_path: str | Path | None = None,
        *,
        mat_path: str | Path | None = None,
        preload: bool = True,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        if data_path is None:
            if mat_path is None:
                raise ValueError("必须提供 data_path 或 mat_path")
            data_path = mat_path
        elif mat_path is not None:
            raise ValueError("data_path 和 mat_path 只能二选一")

        self.data_path = str(Path(data_path))
        self.transform = transform
        self._dtype = dtype

        self._features: Optional[torch.Tensor] = None
        self._labels: Optional[torch.Tensor] = None
        self._length: Optional[int] = None
        self._position_ids: Optional[np.ndarray] = None  # 每个样本的位置ID
        self._metadata: Dict[str, np.ndarray] = {}
        self._attrs: Dict[str, object] = {}

        if preload:
            self._preload_to_memory()

    def _preload_to_memory(self) -> None:
        """从 HDF5 文件加载特征和标签

        项目约定特征张量保存为 (W, H, C, N)，
        读取后转置为 PyTorch 常用的 (N, C, H, W)。
        """
        with h5py.File(self.data_path, "r") as handle:
            features = np.asarray(handle["features"], dtype=self._dtype)  # (W, H, C, N)
            labels = np.asarray(handle["labels"]["position"], dtype=self._dtype)  # (N, D)
            if "metadata" in handle:
                self._metadata = {
                    key: np.asarray(value)
                    for key, value in handle["metadata"].items()
                    if isinstance(value, h5py.Dataset)
                }
            if "config" in handle:
                self._attrs = dict(handle["config"].attrs)

        # 转置 (W, H, C, N) -> (N, C, H, W)
        features = np.transpose(features, (3, 2, 1, 0))

        self._features = torch.from_numpy(features)  # (N, C, H, W)
        if labels.ndim == 1:
            labels = labels[:, None]
        self._labels = torch.from_numpy(labels)  # (N, D)
        self._length = self._features.shape[0]

        # 计算位置ID：将 (x, y, z) 转换为唯一整数ID
        # 用于位置级切分，保证同一位置的样本不跨越 train/val
        self._position_ids = self._compute_position_ids(self._labels)

    def _compute_position_ids(self, labels: torch.Tensor) -> np.ndarray:
        """将位置坐标转换为唯一位置ID

        方法：将坐标映射为毫米精度整数 tuple，再转换为唯一ID
        使用四舍五入到毫米精度（0.001m）避免浮点误差
        """
        rounded = np.round(labels.numpy() * 1000).astype(np.int64)
        _, inverse = np.unique(rounded, axis=0, return_inverse=True)
        return inverse.astype(np.int64)

    def __len__(self) -> int:
        if self._features is None:
            raise RuntimeError("数据集未加载，请先调用 _preload_to_memory")
        assert self._length is not None
        return self._length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._features is None or self._labels is None:
            raise RuntimeError("数据集未加载，请先调用 _preload_to_memory")

        feat = self._features[index]
        lab = self._labels[index]

        tensor = feat.to(torch.float32).clone()
        label = lab.to(torch.float32)

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, label

    def features_tensor(self) -> torch.Tensor:
        """返回特征张量 (N, C, H, W)"""
        if self._features is None:
            self._preload_to_memory()
        assert self._features is not None
        return self._features

    def labels_tensor(self) -> torch.Tensor:
        """返回标签张量 (N, 3)"""
        if self._labels is None:
            self._preload_to_memory()
        assert self._labels is not None
        return self._labels

    def metadata_array(self, name: str) -> np.ndarray | None:
        if self._features is None:
            self._preload_to_memory()
        return self._metadata.get(name)

    def attrs(self) -> Dict[str, object]:
        if self._features is None:
            self._preload_to_memory()
        return dict(self._attrs)

    def position_ids(self) -> np.ndarray:
        """返回位置ID数组 (N,)"""
        if self._position_ids is None:
            self._preload_to_memory()
        assert self._position_ids is not None
        return self._position_ids

    def group_ids(self, strategy: str) -> np.ndarray:
        """Return group ids for leakage-safe splitting."""
        strategy = strategy.lower()
        if strategy == "random":
            return np.arange(len(self), dtype=np.int64)
        if strategy == "ue_level":
            global_mobile = self.metadata_array("global_mobile_id")
            if global_mobile is not None:
                return np.asarray(global_mobile, dtype=np.int64)
            global_tx = self.metadata_array("global_tx_id")
            if global_tx is not None:
                return np.asarray(global_tx, dtype=np.int64)
        if strategy in {"ue_level", "position_level"}:
            return self.position_ids()
        raise ValueError(f"不支持的 split_strategy: {strategy}")

    def unique_positions(self) -> Tuple[np.ndarray, Dict[int, List[int]]]:
        """返回唯一位置ID和每个位置对应的样本索引列表

        Returns:
            unique_ids: 唯一位置ID数组
            position_to_indices: Dict[位置ID -> 样本索引列表]
        """
        pos_ids = self.position_ids()
        unique_ids = np.unique(pos_ids)

        position_to_indices: Dict[int, List[int]] = {}
        for pos_id in unique_ids:
            indices = np.where(pos_ids == pos_id)[0].tolist()
            position_to_indices[pos_id] = indices

        return unique_ids, position_to_indices


def split_indices_by_group(
    group_ids: np.ndarray,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Dict[str, List[int]]:
    total = float(train_fraction) + float(val_fraction) + float(test_fraction)
    if not np.isclose(total, 1.0):
        raise ValueError(f"train/val/test split 之和必须为 1，当前为 {total}")
    unique_groups = np.unique(group_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    n_groups = int(unique_groups.size)
    n_train = int(round(n_groups * train_fraction))
    n_val = int(round(n_groups * val_fraction))
    if val_fraction == 0.0 and test_fraction == 0.0:
        n_train = n_groups
        n_val = 0
    elif val_fraction == 0.0:
        n_train = max(1, min(n_train, n_groups - 1))
        n_val = 0
    elif test_fraction == 0.0:
        n_train = max(1, min(n_train, n_groups - 1))
        n_val = n_groups - n_train
    elif n_groups >= 3:
        n_train = max(1, min(n_train, n_groups - 2))
        n_val = max(1, min(n_val, n_groups - n_train - 1))
    train_groups = set(unique_groups[:n_train].tolist())
    val_groups = set(unique_groups[n_train : n_train + n_val].tolist())
    test_groups = set(unique_groups[n_train + n_val :].tolist())
    splits = {"train": [], "val": [], "test": []}
    for index, group_id in enumerate(group_ids.tolist()):
        if group_id in train_groups:
            splits["train"].append(index)
        elif group_id in val_groups:
            splits["val"].append(index)
        elif group_id in test_groups:
            splits["test"].append(index)
        else:
            raise RuntimeError(f"样本 {index} 未被分配 split")
    return splits
