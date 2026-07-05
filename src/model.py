from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class InputModule(nn.Module):
    """论文版 Input Module: Conv-BN-ReLU-Pool。

    接收 CFR image，通过基础卷积块做初始特征提取，
    把原始图像变成更抽象的 feature map。
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 256, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: (B, C_in, H_in, W_in)
        输出: (B, 256, H_in, W_in/2)
        注意: H 保持不变，W 被池化缩小 2 倍
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class UpperHalf(nn.Module):
    """论文版 Upper Half: 多尺度特征提取器。

    四条并行路径通过 Depth Concatenation 融合为 256 通道：
    - Path 1: BN -> ReLU -> 1×1 conv -> 64 channels
    - Path 2: BN -> ReLU -> 5×1 conv + 1×3 conv -> 32 channels
    - Path 3: MaxPool -> BN -> ReLU -> 3×3 conv -> 32 channels
    - Path 4: BN -> ReLU -> 1×1 conv (32) -> BN -> ReLU -> 1×1 conv (128) -> 128 channels

    总输出: 64 + 32 + 32 + 128 = 256 channels

    入口和出口之间加入 identity mapping 用于稳定训练。
    """

    def __init__(self, in_channels: int = 256, reduced_channels: int = 64) -> None:
        super().__init__()

        # Bottleneck: 将输入压缩到 reduced_channels
        self.bottleneck = nn.Conv2d(in_channels, reduced_channels, kernel_size=(1, 1), bias=False)

        # Path 1: BN -> ReLU -> 1×1 conv -> 64 channels
        self.path1 = nn.Sequential(
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, 64, kernel_size=(1, 1), bias=False),
        )

        # Path 2: BN -> ReLU -> 5×1 conv + 1×3 conv -> 32 channels
        self.path2 = nn.Sequential(
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, 32, kernel_size=(5, 1), padding=(2, 0), bias=False),
            nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1), bias=False),
        )

        # Path 3: MaxPool -> BN -> ReLU -> 3×3 conv -> 32 channels
        self.path3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, 32, kernel_size=(3, 3), padding=1, bias=False),
        )

        # Path 4: BN -> ReLU -> 1×1 conv (32) -> BN -> ReLU -> 1×1 conv (128) -> 128 channels
        self.path4 = nn.Sequential(
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, 32, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, kernel_size=(1, 1), bias=False),
        )

        # Identity mapping / skip connection
        self.skip = nn.Identity() if in_channels == 256 else nn.Conv2d(in_channels, 256, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: (B, 256, H, W)
        输出: (B, 256, H, W)
        注意: 本模块不改变 feature map 的空间尺寸
        """
        residual = self.skip(x)
        x = self.bottleneck(x)
        out = torch.cat(
            [self.path1(x), self.path2(x), self.path3(x), self.path4(x)], dim=1
        )
        out = out + residual  # 残差连接
        return out


class DualPathBasicUnit(nn.Module):
    """论文版双路径基础单元。

    每个 basic unit 包含：
    - path A: BN -> ReLU -> 3×3 conv @256
    - path B: BN -> ReLU -> 1×1 conv @256
    - Cross-connection: 两路径交换特征
    - Residual: 输入直接加到输出

    注意：顺序为 BN -> ReLU -> Conv，与 UpperHalf 的风格一致。
    """

    def __init__(self, channels: int = 256) -> None:
        super().__init__()

        # Path A: BN -> ReLU -> 3×3 conv
        self.bnA = nn.BatchNorm2d(channels)
        self.reluA = nn.ReLU(inplace=True)
        self.convA = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

        # Path B: BN -> ReLU -> 1×1 conv
        self.bnB = nn.BatchNorm2d(channels)
        self.reluB = nn.ReLU(inplace=True)
        self.convB = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, a_in: torch.Tensor, b_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输入: a_in, b_in 各为 (B, 256, H, W)
        输出: a_out, b_out 各为 (B, 256, H, W)
        注意: 本单元不改变空间尺寸

        处理顺序: BN -> ReLU -> Conv（pre-activation 风格）
        Cross-connection + Residual:
        - a_out = a_in + convA(bnA(reluA(a_in))) + convB(bnB(reluB(b_in)))
        - b_out = b_in + convB(bnB(reluB(b_in))) + convA(bnA(reluA(a_in)))
        """
        # BN -> ReLU -> Conv (pre-activation 风格)
        a_conv = self.convA(self.reluA(self.bnA(a_in)))
        b_conv = self.convB(self.reluB(self.bnB(b_in)))

        # Cross-connection + Residual
        a_out = a_in + a_conv + b_conv
        b_out = b_in + b_conv + a_conv

        return a_out, b_out


class LowerHalf(nn.Module):
    """论文版 Lower Half: 3-stage 双路径特征传播与坐标映射。

    结构：
    - 共 3 个 stage，每个 stage 包含 1 个 DualPathBasicUnit
    - stage 1: 无下采样
    - stage 2: 入口处 MaxPool 下采样 (stride=2)
    - stage 3: 入口处 MaxPool 下采样 (stride=2)

    最终两条路径特征相加融合。
    """

    def __init__(self, channels: int = 256, num_stages: int = 3) -> None:
        super().__init__()

        assert num_stages == 3, "论文版 LowerHalf 固定为 3 个 stage"

        self.num_stages = num_stages

        # 每个 stage 的下采样池化（只在 stage 2 和 3 入口处使用）
        self.stage_downsample = nn.ModuleList([
            None,  # stage 1: 无下采样
            nn.MaxPool2d(kernel_size=2, stride=2),  # stage 2: 下采样
            nn.MaxPool2d(kernel_size=2, stride=2),  # stage 3: 下采样
        ])

        # 每个 stage 包含 1 个 dual-path basic unit
        # 共 3 个 dual-path blocks
        self.units = nn.ModuleList([
            DualPathBasicUnit(channels) for _ in range(num_stages)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: (B, 256, H, W)
        输出: (B, 256, H_out, W_out)
        注意: 若 stage 2 和 stage 3 都做 stride=2 下采样，则输出相对输入缩小 4 倍
        """
        a = x  # path A 初始化
        b = x  # path B 初始化（两条路径从同一输入开始）

        for stage in range(self.num_stages):
            # Stage 入口下采样（只对 stage 2 和 3）
            if self.stage_downsample[stage] is not None:
                pool = self.stage_downsample[stage]
                a = pool(a)
                b = pool(b)

            # 每个 stage 只执行 1 个 dual-path basic unit
            a, b = self.units[stage](a, b)

        # 两条路径融合
        out = a + b
        return out


class OutputModule(nn.Module):
    """论文版 Output Module: 从 feature 到坐标输出。

    结构：
    - Conv-BN-ReLU 做特征整理
    - AdaptiveMaxPool 将特征压缩到全局
    - Dropout 抑制过拟合
    - Linear 输出三维坐标 (x, y, z)
    """

    def __init__(self, in_channels: int = 256, dropout: float = 0.3, output_dim: int = 2) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((1, 1)),  # -> (B, C, 1, 1)
            nn.Flatten(1),                 # -> (B, C)
            nn.Dropout(dropout),
            nn.Linear(in_channels, output_dim),     # -> (B, D)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: (B, 256, H, W)
        输出: (B, D)
        """
        return self.head(x)


class MPRINet(nn.Module):
    """论文版 MPRI 网络。

    整体流程：
    CFR image -> Input Module -> Upper Half -> Lower Half -> Output Module -> UE Position (x, y, z)

    各模块职责：
    - InputModule: 初始特征提取，宽度下采样 2 倍
    - UpperHalf: 多尺度特征提取，不改变空间尺寸
    - LowerHalf: 3-stage 双路径传播，高度宽度各下采样 4 倍（共 2 次 stride=2）
    - OutputModule: 特征整理与坐标输出
    """

    def __init__(
        self,
        in_channels: int = 2,
        dropout: float = 0.3,
        output_dim: int = 2,
    ) -> None:
        super().__init__()

        self.input_module = InputModule(in_channels)
        self.mpri_upper = UpperHalf(256, 64)  # 输入256通道，输出256通道
        self.mpri_lower = LowerHalf(channels=256, num_stages=3)  # 论文版固定3-stage
        self.output_module = OutputModule(256, dropout=dropout, output_dim=output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """权重初始化：Conv用Kaiming，BN用ones/zeros，Linear用Xavier"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: (B, C_in, H_in, W_in)
        输出: (B, 3)
        """
        x = self.input_module(x)
        x = self.mpri_upper(x)
        x = self.mpri_lower(x)
        x = self.output_module(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """计算模型可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型结构和参数量
    model = MPRINet(in_channels=2)
    print(f"模型参数量: {count_parameters(model):,}")

    # 测试前向传播
    dummy_input = torch.randn(4, 2, 100, 10)  # (B, C, H, W)
    output = model(dummy_input)
    print(f"输入 shape: {dummy_input.shape}")
    print(f"输出 shape: {output.shape}")
