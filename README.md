# MPRI Baseline

This project adapts the MPRI CFR-image localization baseline to the current baseline matrix data scope:

- Front3D CFR-only HDF5: `Front3d_2988_CFR_Only`
- DeepMIMO O1_3p5 path-parameter `.mat`

The active conversion path writes the MPRI input contract:

```text
features: (W, H, C, N)
labels/position: (N, 2)
```

`W` is the flattened CFR/image feature width, `H` is selected subcarriers, `C` is the BS/anchor depth, and `N` is samples. Configs must declare `dataset.format: front3d` or `dataset.format: deepmimo`. Data roots can be overridden with `BASELINE_FRONT3D_ROOT` and `BASELINE_DEEPMIMO_ROOT`.

## Paper Alignment

Confirmed from the paper:

- CFR images are built by flattening MIMO CFR observations.
- Noise augmentation uses three SNR levels shown/described as `50 dB`, `20 dB`, and `10 dB`.
- The paper adds noise at the PHY `Filter_multipath` stage; this repo applies AWGN to available CFR or path-derived CFR as an adaptation because the current datasets do not store the original PHY filter.
- Training uses `400` epochs, initial learning rate `1e-3`, and a `50/0.3` epoch/drop-factor schedule.
- The paper reports a `0.8` train/test split.

Not fully specified in the paper and still tracked for review: optimizer details, batch size, and the exact DeepMIMO path-parameter-to-CFR synthesis contract.

## Environment

```bash
uv sync --extra dev
```

If the PyTorch index is unreachable, the existing project venv can still be used after installing SciPy:

```bash
uv pip install --python .venv/bin/python -i https://pypi.tuna.tsinghua.edu.cn/simple "scipy>=1.15"
```

## Data Check And Conversion

Front3D smoke:

```bash
uv run python scripts/check_data.py --config config/data/front3d_5431_smoke.yaml
uv run python scripts/convert_cfr_to_csi.py --config config/data/front3d_5431_smoke.yaml --overwrite
```

DeepMIMO smoke:

```bash
uv run python scripts/check_data.py --config config/data/deepmimo_o1_3p5_smoke.yaml
uv run python scripts/convert_cfr_to_csi.py --config config/data/deepmimo_o1_3p5_smoke.yaml --overwrite
```

Paper-style Front3D config:

```bash
uv run python scripts/convert_cfr_to_csi.py --config config/data/front3d_5431_paper.yaml --overwrite
uv run python train_mpri.py --config config/train/front3d_5431_paper400.yaml
```

## Smoke Train And Eval

```bash
uv run python train_mpri.py --config config/train/smoke.yaml
uv run python evaluate_mpri.py \
  --config config/eval/default.yaml \
  --checkpoint runs/train/<front3d-run>/checkpoints/best.pt
```

DeepMIMO smoke:

```bash
uv run python train_mpri.py --config config/train/deepmimo_o1_3p5_smoke.yaml
uv run python evaluate_mpri.py \
  --config config/eval/deepmimo_o1_3p5_smoke.yaml \
  --checkpoint runs/train/<deepmimo-run>/checkpoints/best.pt
```

Each training run writes `config.yaml`, `train.log`, `metrics.csv`, TensorBoard logs, and `checkpoints/best.pt` / `last.pt` under `runs/train/`.

## Tests

```bash
uv run python -m pytest -q
```
