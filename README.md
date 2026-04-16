# MPRI 项目（论文 Baseline 复现）

MPRI（Multi-Path Residual Inference）网络用于基于 CSI/CFR 信号的室内定位，恢复与论文一致的 baseline。

当前仓库使用 Sionna 仿真输出的 CFR 数据，通过数据适配器统一转换为 MPRI 可训练/评估的 HDF5 数据格式。

## 项目结构

```
MPRI/
├── src/
│   ├── model.py          # MPRI 网络定义
│   └── data.py           # 数据集加载与归一化
├── scripts/
│   └── convert_cfr_to_csi.py  # Sionna CFR -> MPRI CSI 数据适配器
├── data/
│   ├── raw/              # 原始 Sionna CFR 数据（.h5）
│   ├── csi/              # 处理后 MPRI 数据（.h5）
│   └── temp/             # 可选临时目录
├── train_mpri.py         # 训练入口
├── evaluate_mpri.py      # 评估入口
├── checkpoints/          # 模型权重（运行时生成）
├── plots/                # 损失曲线（运行时生成）
└── results/              # 预测结果（运行时生成）
```

## 输入格式

模型输入为 CSI 矩阵 `(B, C, H, W)`：

| 维度 | 含义 |
|------|------|
| B | batch size |
| C | AP / gNB 数 (Ng)，自动从输入数据适配 |
| H | 子载波数 (Ns)，自动从输入数据适配 |
| W | MIMO 通道数 (Np)，`num_tx_ant * num_rx_ant * 2` |

W 维度采用幅度-相位交织排列：`[amp₁, phase₁, amp₂, phase₂, ..., amp₁₆, phase₁₆]`

## 环境配置

```bash
conda env create -f environment_final.yml
conda activate mpri
```

## 数据预处理

原始数据放在 `data/raw/`，运行数据适配器转换为模型可直接读取的 `h5` 文件：

```bash
python scripts/convert_cfr_to_csi.py
```

默认行为：
- 输入路径：`data/raw/`（目录下需匹配到 1 个 `.h5` 文件）
- 输出目录：`data/csi/`

脚本会生成两个策略文件（AP 数自动适配，不固定）：

| 输出文件 | 说明 | 样本数 |
|----------|------|--------|
| `data/csi/csi_strict.h5` | 仅保留所有 AP 都有信号的位置 | 由数据决定 |
| `data/csi/csi_padded.h5` | 至少一个 AP 有信号，缺失 AP 零填充 | 由数据决定 |

常用参数示例：

```bash
python scripts/convert_cfr_to_csi.py \
  --input data/raw/test5.h5 \
  --output-dir data/csi \
  --chunk-size 512 \
  --overwrite
```

## 训练

```bash
python train_mpri.py \
  --dataset data/csi/csi_strict.h5 \
  --epochs 100 \
  --batch-size 8 \
  --amp \
  --device cuda \
  --save-path checkpoints/model.pt \
  --loss-figure plots/loss_curve.png
```

关键参数：
- `--amp`：混合精度训练，减少显存占用（RTX 3050 Ti 等 4GB 显存必须开启）
- `--dataset`：数据集路径，传入转换后的 `csi_*.h5`
- `--epochs`：训练轮数（论文默认 400）
- `--batch-size`：批大小

## 评估

```bash
python evaluate_mpri.py \
  --checkpoint checkpoints/model.pt \
  --dataset data/csi/csi_strict.h5 \
  --split val \
  --batch-size 64
```


## TODO
明确一下 data 下的存放规范，并修改转换脚本和读取方式：
1. raw 和 csi 下面再加一个文件夹以区分不同的数据
2. 再补充一个 json 的元数据清单，有数据规模等等信息