# MPRI Baseline

This project adapts the MPRI CFR-image localization baseline to the current baseline matrix data scope:

- Front3D CFR-only HDF5: `Front3d_2988_CFR_Only`
- DeepMIMO offline CFR HDF5 produced by `../DeepMIMOFactory`

The active conversion path writes the MPRI input contract:

```text
features: (W, H, C, N)
labels/position: (N, 2)
```

`W` is the flattened CFR/image feature width, `H` is selected subcarriers, `C` is the BS/anchor depth, and `N` is samples. Configs must declare `dataset.format: front3d` or `dataset.format: deepmimo`. Data roots can be overridden with `BASELINE_FRONT3D_ROOT` and `BASELINE_DEEPMIMO_CFR_ROOT`.

DeepMIMO production is intentionally outside this repository. MPRI consumes `deepmimo_cfr_v1` HDF5 scenes with `cfr` axes `[link, rx_ant, tx_ant, subcarrier]` and then performs its own MPRI conversion plus optional SNR augmentation. It does not read DeepMIMO raw `.mat` files or synthesize CFR from path parameters.

## Paper Alignment

Confirmed from the paper:

- CFR images are built by flattening MIMO CFR observations.
- Noise augmentation uses three SNR levels shown/described as `50 dB`, `20 dB`, and `10 dB`.
- The paper adds noise at the PHY `Filter_multipath` stage; this repo applies AWGN to available offline CFR as an adaptation because the current datasets do not store the original PHY filter.
- Training uses `400` epochs, initial learning rate `1e-3`, and a `50/0.3` epoch/drop-factor schedule.
- The paper reports a `0.8` train/test split.

Not fully specified in the paper and still tracked for review: optimizer details and batch size.

## Front3D Noisy Workflow

The paper uses SNR augmentation at `50/20/10 dB`. For Front3D, the current reproducible adaptation is an offline noisy dataset that expands each CFR link once per SNR level:

```bash
BASELINE_FRONT3D_ROOT=/path/to/Front3d_2988_CFR_Only \
.venv/bin/python scripts/build_front3d_noise_subset.py --chunk-links 1024
```

By default, the script writes to `$BASELINE_FRONT3D_ROOT/noise_sub_test_set`. Override with `BASELINE_FRONT3D_NOISE_ROOT` or `--output-root`. The generated `manifest.json` is authoritative only after the builder completes all selected scenes; interrupted `.h5.tmp` files are not part of the dataset contract.

DeepMIMO smoke uses the same converter with `dataset.format: deepmimo`; the input must already be offline CFR from `DeepMIMOFactory`:

```bash
BASELINE_DEEPMIMO_CFR_ROOT=/path/to/deepmimo_cfr_root \
.venv/bin/python scripts/convert_cfr_to_csi.py \
  --config config/data/deepmimo_o1_3p5_smoke.yaml --overwrite
```

The current DeepMIMO smoke source is the axis-corrected Factory export with `cfr.shape == (16, 16, 2, 768)`. The MPRI smoke config selects 64 subcarriers and writes 24 samples from 8 UE x 3 SNR levels.

## BS Convention

MPRI consumes converted scene-level MPRI HDF5 files. The converter supports two formal BS modes through `front3d` config:

- Single BS: set `bs_selection: random`, `bs_count: 1`, and `bs_seed`; the converter picks one deterministic random scene representative BS.
- Multi BS: set `bs_selection: random`, `bs_count: 4`, and `bs_seed`; the converter picks four deterministic random BS for the scene. The padded output takes the union of UE positions observed by the selected BS and fills missing BS feature slots with zero. The strict output keeps only UE samples observed by all selected BS.

`random` is the converter default for Front3D/DeepMIMOFactory CFR inputs. Explicit `bs_ids` still overrides random selection. For formal multi-BS baseline runs, use the padded output unless the experiment intentionally requires all four BS to observe every UE.

## Metrics And Outputs

Previous local MPRI test results are treated as obsolete after the metric/BS convention update and should be regenerated before reporting.

Each conversion run writes under `data/processed/`; each training run writes `config.yaml`, `train.log`, `metrics.csv`, TensorBoard logs, and `checkpoints/best.pt` / `last.pt` under `runs/train/`. These directories are intentionally ignored by git. Because the paper-style config sets `val_split=0` and evaluates test each epoch, `best.pt` is selected by test RMSE. Use `last.pt` when reporting a non-test-selected endpoint.

Unified evaluation metrics are `mean_error_m`, `mse_m2`, `median_error_m`, `rmse_m`, `cdf_1m`, `cdf_2m`, `p90_error_m`, `p95_error_m`, `p99_error_m`, and `valid_sample_count`. `evaluate_mpri.py` writes this schema to `metrics.csv`; `train_mpri.py` keeps the old epoch columns such as `train_rmse` and `train_mean_err` and also adds the prefixed unified columns for train/val/test.

`predictions.csv` preserves per-sample results. Single-BS converted data is reported as `sample_kind=ue_bs_pair`; multi-BS converted data is reported as `sample_kind=ue_bs_set`. The table includes `bs_set_ids`, `valid_bs_ids`, `valid_bs_count`, and `source_link_ids`; missing selected BS slots are visible through `valid_bs_ids` and `source_link_ids`.
