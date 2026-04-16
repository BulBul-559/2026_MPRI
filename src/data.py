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

        if preload:
            self._preload_to_memory()

    def _preload_to_memory(self) -> None:
        """从 HDF5 文件加载特征和标签

        项目约定特征张量保存为 (W, H, C, N)，
        读取后转置为 PyTorch 常用的 (N, C, H, W)。
        """
        with h5py.File(self.data_path, "r") as handle:
            features = np.asarray(handle["features"], dtype=self._dtype)  # (W, H, C, N)
            labels = np.asarray(handle["labels"]["position"], dtype=self._dtype)  # (N, 3)

        # 转置 (W, H, C, N) -> (N, C, H, W)
        features = np.transpose(features, (3, 2, 1, 0))

        self._features = torch.from_numpy(features)  # (N, C, H, W)
        self._labels = torch.from_numpy(labels)  # (N, 3)
        self._length = self._features.shape[0]

        # 计算位置ID：将 (x, y, z) 转换为唯一整数ID
        # 用于位置级切分，保证同一位置的样本不跨越 train/val
        self._position_ids = self._compute_position_ids(self._labels)

    def _compute_position_ids(self, labels: torch.Tensor) -> np.ndarray:
        """将位置坐标转换为唯一位置ID

        方法：将 (x, y, z) 三维坐标映射为整数 tuple，再转换为唯一ID
        使用四舍五入到毫米精度（0.001m）避免浮点误差
        """
        # 四舍五入到毫米精度
        rounded = np.round(labels.numpy() * 1000).astype(np.int64)
        # 将三维坐标转换为唯一ID
        # 使用简单的 hash: x * 10^9 + y * 10^6 + z
        position_ids = (
            rounded[:, 0] * 1_000_000_000 +
            rounded[:, 1] * 1_000_000 +
            rounded[:, 2]
        )
        return position_ids

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

    def position_ids(self) -> np.ndarray:
        """返回位置ID数组 (N,)"""
        if self._position_ids is None:
            self._preload_to_memory()
        assert self._position_ids is not None
        return self._position_ids

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