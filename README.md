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

## Front3D Noisy Workflow

The paper uses SNR augmentation at `50/20/10 dB`. For Front3D, the current reproducible adaptation is an offline noisy dataset that expands each CFR link once per SNR level:

```bash
BASELINE_FRONT3D_ROOT=/path/to/Front3d_2988_CFR_Only \
.venv/bin/python scripts/build_front3d_noise_subset.py --chunk-links 1024
```

By default, the script writes to `$BASELINE_FRONT3D_ROOT/noise_sub_test_set`. Override with `BASELINE_FRONT3D_NOISE_ROOT` or `--output-root`. The generated `manifest.json` is authoritative only after the builder completes all selected scenes; interrupted `.h5.tmp` files are not part of the dataset contract.

## Single-Scene 400-Epoch Evidence

The current local paper-style smoke/formal check uses one prepared noisy scene, `front3d_0015`, with `400` epochs and the paper-style LR schedule. These `runs/` artifacts are local and are not committed.

| Input | Best checkpoint test mean / median / P95 / RMSE | Last checkpoint test mean / median / P95 / RMSE | Training time |
|---|---|---|---|
| 193 subcarriers | `0.572 / 0.480 / 1.279 / 0.690 m` at epoch 25 | `0.577 / 0.425 / 1.573 / 0.817 m` at epoch 400 | `6:47.28` |
| Raw 768 subcarriers | `0.558 / 0.458 / 1.333 / 0.687 m` at epoch 26 | `0.603 / 0.449 / 1.522 / 0.812 m` at epoch 400 | `21:52.69` |

Because the paper-style config sets `val_split=0` and evaluates test each epoch, `best.pt` is selected by test RMSE. Use `last.pt` when reporting a non-test-selected endpoint.

Each conversion run writes under `data/processed/`; each training run writes `config.yaml`, `train.log`, `metrics.csv`, TensorBoard logs, and `checkpoints/best.pt` / `last.pt` under `runs/train/`. These directories are intentionally ignored by git.
