# MPRI Baseline

This project adapts the MPRI CFR-image localization baseline to the current baseline matrix data scope:

- Front3D CFR-only HDF5: `Front3d_2988_CFR_Only`
- DeepMIMO offline CFR HDF5 produced by `../DeepMIMOFactory`
- DICHASUS dcxx measured CFR through the locked VisionRadioLoc DXT manifests

The active conversion path writes the MPRI input contract:

```text
features: (W, H, C, N)
labels/position: (N, D)
```

`W` is the flattened CFR/image feature width, `H` is selected subcarriers, `C` is the BS/anchor depth, and `N` is samples. Configs must declare `dataset.format: front3d` or `dataset.format: deepmimo`. Data roots can be overridden with `BASELINE_FRONT3D_ROOT`, `BASELINE_FRONT3D_NOISE_ROOT`, and `BASELINE_DEEPMIMO_CFR_ROOT`.

`D=2` for the current Front3D/DeepMIMO adaptations. The original paper indoor
dataset uses `D=3`.

DeepMIMO production is intentionally outside this repository. MPRI consumes `deepmimo_cfr_v1` HDF5 scenes with `cfr` axes `[link, rx_ant, tx_ant, subcarrier]` and then performs its own MPRI conversion plus optional SNR augmentation. It does not read DeepMIMO raw `.mat` files or synthesize CFR from path parameters.

## Formal Front3D Standard

Formal Front3D single-BS MPRI runs are generated from the baseline scheduler,
not from the legacy paper-style example configs. The current reporting standard is:

- Dataset: `Front3d_2988_CFR_Only/noise_sub_test_set`
- Split protocol: `m117_single_scene_ue_level`, one independent task per test-scene-pool scene
- Split source: `train_links.pt`, `val_links.pt`, and `test_links.pt` under each scene's m117 UE-level split directory
- Link eligibility: the m117 link plan is the source of truth; conversion must not admit raw HDF5 links based only on `valid_mask`
- BS mode: read `single_bs_ids` or `multi_bs_ids` from the canonical `bs_selection_v1` contract
- Subcarriers: raw 768 with `subcarriers.policy: first`
- Training: 60 epochs on `cuda:0`
- Selection/eval: `best.pt` selected by validation RMSE, then `evaluate_mpri.py` on the scene `test` split

Generate and run the formal 339-scene queue from `D:\SunMeiyuan\baseline`:

```powershell
.\2026_MPRI\.venv\Scripts\python.exe .\scheduler\new_mpri_front3d_singlebs_queue.py
.\scheduler\Start-BaselineQueue.ps1 -QueueManifest <generated-manifest> -ContinueOnError
```

## Paper Alignment

Confirmed from the paper:

- CFR images are built by flattening MIMO CFR observations.
- Noise augmentation uses three SNR levels shown/described as `50 dB`, `20 dB`, and `10 dB`.
- The paper adds noise at the PHY `Filter_multipath` stage; this repo applies AWGN to available offline CFR as an adaptation because the current datasets do not store the original PHY filter.
- The paper training schedule uses `400` epochs, initial learning rate `1e-3`, and a `50/0.3` epoch/drop-factor schedule. The formal Front3D standard above uses 60 epochs instead of the paper-style 400-epoch budget.
- The paper reports a `0.8` train/test split.

Not fully specified in the paper and still tracked for review: optimizer details and batch size.

## Original Paper Indoor Dataset

The original Scenario 1 MATLAB v7.3 files are supported through a dedicated
adapter rather than the Front3D/DeepMIMO CFR converter:

```bash
MPRI_PAPER_INDOOR_ROOT=/path/to/mpri_raw \
.venv/bin/python scripts/convert_paper_indoor_data.py \
  --config config/data/paper_indoor_source.yaml
```

The adapter requires the author-provided `50/20/10 dB` files to contain
identical `4816 x 3` position labels and CFR images with h5py axes
`(sample, gNB, MIMO channel, subcarrier)` and shape `(4816, 5, 16, 193)`. It writes
`data/processed/paper_indoor_source/csi_snr50_20_10.h5` with project storage
shape `(16, 193, 5, 14448)`, which the loader exposes as
`(14448, 5, 193, 16)`. No second AWGN pass is applied: these files already
contain the paper's PHY-stage noise variants.

The formal source-data run uses `config/train/paper_indoor_source_paper400.yaml`:

- UE-grouped `80/0/20` split, yielding the paper counts `11559/0/2889`
- all three SNR variants of a UE remain in one split
- 3D position head and five-gNB input
- no additional input standardization, because the released CFR images are
  already processed and the paper does not state a second standardization step
- 400 epochs, `1e-3` initial learning rate, and `StepLR(50, 0.3)`
- test evaluation every epoch and test-RMSE best-checkpoint selection, matching
  the paper's stated experimental protocol

The paper does not publish its split seed, optimizer, or batch size. This
repository fixes seed `2026`, Adam, batch size `32`, weight decay `0`, and
dropout `0.3` as explicit implementation choices. The full audit and run record
are in `docs/paper_indoor_source_reproduction_2026-07-23.md`.

The completed paper400 run produced a test mean error of `2.1146 m`, median
error of `1.6768 m`, and RMSE of `2.6880 m` over all `2889` test images. This
validates the source-data training/evaluation chain, but it does not reproduce
the paper's reported `0.28 m` indoor mean error. The result must therefore be
described as a paper-aligned reimplementation result, not as a successful
numerical reproduction of the paper.

## Front3D Noisy Workflow

The paper uses SNR augmentation at `50/20/10 dB`. For Front3D, the current reproducible adaptation is an offline noisy dataset that expands each CFR link once per SNR level:

```bash
BASELINE_FRONT3D_ROOT=/path/to/Front3d_2988_CFR_Only \
.venv/bin/python scripts/build_front3d_noise_subset.py --chunk-links 1024
```

By default, the script writes to `$BASELINE_FRONT3D_ROOT/noise_sub_test_set`. Override with `BASELINE_FRONT3D_NOISE_ROOT` or `--output-root`. The generated `manifest.json` is authoritative only after the builder completes all selected scenes; interrupted `.h5.tmp` files are not part of the dataset contract.

The current local dataset expects the complete `m117_seed2026_scene_level` noisy test set:

- `339` completed test scenes
- `934,128` source links expanded to `2,802,384` noisy links
- SNR levels `50/20/10 dB`

Set both roots explicitly before conversion:

```powershell
$env:BASELINE_FRONT3D_ROOT = "D:\SunMeiyuan\datasets\Front3d_2988_CFR_Only"
$env:BASELINE_FRONT3D_NOISE_ROOT = "$env:BASELINE_FRONT3D_ROOT\noise_sub_test_set"
```

Use the current smoke config to verify the active dataset wiring. It reads `splits/m117_seed2026_scene_level/test_scenes.txt`, takes the first scene with `scene_limit: 1`, and checks `build_status.csv` before touching the HDF5 data:

```powershell
python scripts/check_data.py --config config/data/front3d_noise_sub_current_smoke.yaml
python scripts/convert_cfr_to_csi.py --config config/data/front3d_noise_sub_current_smoke.yaml --overwrite
python train_mpri.py --config config/train/front3d_noise_sub_current_smoke.yaml
```

Legacy paper-style single-scene configs are still available for reproducibility,
including `front3d_noise_sub_current_paper400.yaml` and
`front3d_noise_sub_0015_paper400.yaml`. They are development/audit configs, not
the formal Front3D reporting route. Formal collection-level reporting should
run scenes separately through the scheduler and aggregate metrics/predictions
after all selected scene tasks finish.

DeepMIMO smoke uses the same converter with `dataset.format: deepmimo`; the input must already be offline CFR from `DeepMIMOFactory`:

```bash
BASELINE_DEEPMIMO_CFR_ROOT=/path/to/deepmimo_cfr_root \
.venv/bin/python scripts/convert_cfr_to_csi.py \
  --config config/data/deepmimo_o1_3p5_smoke.yaml --overwrite
```

The current DeepMIMO smoke source is the axis-corrected Factory export with `cfr.shape == (16, 16, 2, 768)`. The MPRI smoke config selects 64 subcarriers and writes 24 samples from 8 UE x 3 SNR levels.

## BS Convention

MPRI consumes converted scene-level MPRI HDF5 files. The converter supports two
BS modes through `front3d` config:

- Single BS: use the per-scene `single_bs_ids` from the canonical contract.
- Multi BS: use the per-scene `multi_bs_ids` from the canonical contract.

The current Front3D M117 contract is:

```text
baseline/contracts/front3d/Front3d_2988_CFR_Only/m117_single_scene_ue_level/bs_selection_v1/
```

It was derived from the current MPRI deterministic-random convention and then
frozen so every model uses the same BS entrypoint. The generator remains
reproducible (`bs_selection: random`, `bs_seed: 2026`, plus the scene-id hash),
but new baselines should read the contract instead of independently selecting
BS IDs. Explicit `bs_ids` overrides automatic selection in the converter.

For formal Front3D M117 conversion, the converter must receive
`dataset.split_plan_dir` or `front3d.split_plan_dir`. It filters source HDF5
rows by the union of `train_links.pt`, `val_links.pt`, and `test_links.pt`.
For the noisy Front3D dataset this filter is applied through
`noise/source_local_link_index`, so expanded SNR rows still inherit the clean
m117 link eligibility. A converted HDF5 containing a UE or source link outside
the split plan is invalid and must be regenerated.

For formal multi-BS baseline runs, use the padded output. Padded samples keep
the fixed per-scene `bs_set_ids`, fill missing selected-BS feature slots with
zero, and expose the per-sample available subset as `valid_bs_ids`. Strict
output keeps only UE samples observed by all selected BS and is reserved for
explicit ablations.

For the current Front3D test scene pool, MPRI must use the predefined per-scene UE split plan under `m117_single_scene_ue_level`. Training configs can set `data.split_plan_dir` or pass `--split-plan-dir`; when present, `train_mpri.py` uses that split plan instead of creating a second random split inside the converted HDF5. This training split is a second guard; the conversion stage must already have applied the same m117 link plan filter.

## Metrics And Outputs

Previous local MPRI test results are treated as obsolete after the metric/BS convention update and should be regenerated before reporting.

Each conversion run writes under `data/processed/`; each training run writes `config.yaml`, `train.log`, `metrics.csv`, TensorBoard logs, and `checkpoints/best.pt` / `last.pt` under `runs/train/`. These directories are intentionally ignored by git. Formal scene-pool configs use the predefined train/val/test split plan and leave the test split for final inference with `evaluate_mpri.py`.

Unified evaluation metrics are `mean_error_m`, `mse_m2`, `median_error_m`, `rmse_m`, `cdf_1m`, `cdf_2m`, `p90_error_m`, `p95_error_m`, `p99_error_m`, and `valid_sample_count`. `evaluate_mpri.py` writes this schema to `metrics.csv`; `train_mpri.py` keeps the old epoch columns such as `train_rmse` and `train_mean_err` and also adds the prefixed unified columns for train/val/test.

`predictions.csv` preserves per-sample results. Single-BS converted data is reported as `sample_kind=ue_bs_pair`; multi-BS converted data is reported as `sample_kind=ue_bs_set`. The table includes `bs_set_ids`, `valid_bs_ids`, `valid_bs_count`, and `source_link_ids`; missing selected BS slots are visible through `valid_bs_ids` and `source_link_ids`. Datasets with SNR variants also export `snr_db`, `source_position_index`, and `source_variant_index`; their evaluation run includes `metrics_by_snr.csv`.

Collection-level reporting uses the same two BaselineAnalysis aggregation routes: average scene-level `metrics.csv` rows for scene-mean metrics, and concatenate all scene `predictions.csv.error_m` rows to recompute pooled metrics/CDF. CDF curves are always pooled-sample CDFs over the full test scene pool.

The formal DICHASUS entry point is `scripts/run_dichasus_formal.py`. It keeps MPRI's 193 uniformly sampled subcarriers, `50/20/10 dB` training augmentation, 60-epoch schedule, and protocol-declared validation-RMSE checkpoint selection while reading evaluation CFR in chunks. It supports both the nested DXT label-budget manifests and arbitrary positive full-train/dc40 `8:1:1` training sizes. Evaluation cohorts are declared by the VisionRadioLoc protocol. Every final query is written to `predictions.csv` with run/measurement/BS/H5-row/link identity, prediction, GT, and radial error. `cost.json` records optimizer steps, augmented sample consumption, trainable parameters, wall-clock, and target-label count.
