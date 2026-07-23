# MPRI Original Indoor Source-Data Reproduction

Date: 2026-07-23

## Purpose

Train and test the current paper-aligned MPRI implementation on the original
paper's released Scenario 1 indoor data, instead of an adapted Front3D or
DeepMIMO collection.

## Source Contract

Local source root:

```text
/data/sunmeiyuan/projects/BaselineAnalysis/data/mpri_raw
```

| SNR | Source file | h5py feature shape `(sample,gNB,MIMO,subcarrier)` | Label shape |
|---:|---|---|---|
| 50 dB | `dataset_SNR50_indoor_21-11-16_23-11.mat` | `(4816, 5, 16, 193)` | `(4816, 3)` |
| 20 dB | `dataset_SNR20_indoor_21-11-17_01-50.mat` | `(4816, 5, 16, 193)` | `(4816, 3)` |
| 10 dB | `dataset_SNR10_indoor_21-11-17_03-07.mat` | `(4816, 5, 16, 193)` | `(4816, 3)` |

MATLAB v7.3 reverses the physical HDF5 dimension order. The corresponding
paper/MATLAB shape is `193 x 16 x 5 x 4816`:

- 193 frequency samples
- 16 flattened MIMO channels
- 5 gNBs
- 4816 UE positions

The three label arrays are exactly equal. All 4816 labels are unique. Coordinate
ranges are:

| Axis | Min | Max |
|---|---:|---:|
| x | 4.4 m | 58.4 m |
| y | 4.3 m | 16.3 m |
| z | 1.2 m | 1.7 m |

The released files already contain noise inserted by the authors' PHY
`Filter_multipath` pipeline. This reproduction does not add offline CFR noise
again.

## Conversion

Config: `config/data/paper_indoor_source.yaml`

Command:

```bash
.venv/bin/python scripts/convert_paper_indoor_data.py \
  --config config/data/paper_indoor_source.yaml
```

Generated local artifact:

```text
data/processed/paper_indoor_source/csi_snr50_20_10.h5
```

Verified contract:

- storage shape: `(16, 193, 5, 14448)` as `(W,H,C,N)`
- loader shape: `(14448, 5, 193, 16)` as `(N,C,H,W)`
- dtype: float32
- file size: about 670 MiB
- conversion wall time: 41.96 s
- conversion peak RSS: about 96 MiB
- fixed sample order: SNR variant major, then source position
- sampled raw-to-converted maximum absolute difference after float32 cast: 0

The adapter writes `global_mobile_id`, `source_position_index`,
`source_variant_index`, `snr_db`, `source_file`, five anchor IDs, and a complete
anchor-validity mask.

## Split Contract

The paper reports 14448 indoor CFR images, with 11559 for training and 2889 for
testing. These counts factor exactly as:

```text
3853 train UE x 3 SNR variants = 11559
 963 test UE  x 3 SNR variants =  2889
```

The implementation therefore splits 4816 UE groups first and expands all three
SNR variants within each assigned split. Seed `2026` is fixed because the paper
does not publish its split seed. The train/test UE overlap is verified to be
zero.

## Training And Evaluation Contract

Training config: `config/train/paper_indoor_source_paper400.yaml`

Evaluation config: `config/eval/paper_indoor_source_paper400.yaml`

| Item | Value | Basis |
|---|---:|---|
| Epochs | 400 | Paper |
| Initial LR | 1e-3 | Paper |
| LR schedule | StepLR, 50 epochs, gamma 0.3 | Paper |
| Train/test | 80/20, no validation | Paper |
| Test each epoch | Yes | Paper |
| Best checkpoint | Test RMSE | Paper-style protocol |
| Input | 5 x 193 x 16 | Released data and paper `Ns x Nc x Ng` semantics |
| Output | 3D | Released labels/paper |
| Extra normalization | None | Paper does not specify it; released images are already processed |
| Optimizer | Adam | Explicit local choice; paper unspecified |
| Batch size | 32 | Explicit local choice; paper unspecified |
| Weight decay | 0 | Explicit local choice; paper unspecified |
| Seed | 2026 | Explicit local choice; paper unspecified |

The test-selected `best.pt` is the paper-style headline checkpoint. `last.pt`
remains available as the non-selection endpoint.

## Smoke Evidence

The smoke uses the full 11559/2889 split and only shortens training to one
epoch.

- verified input: `5 x 193 x 16`
- model parameters: 2,631,235
- epoch wall time, including per-epoch test: 23.9 s
- end-to-end process wall time: 48.32 s
- process peak RSS: about 2.20 GiB
- epoch-1 test mean error: 8.6224 m
- epoch-1 test RMSE: 9.9806 m
- independent evaluation count: 2889
- all three per-SNR groups contain exactly 963 samples
- full repository tests after adaptation: 18 passed

This is only a chain/resource gate and is not a paper result. An earlier
`5 x 16 x 193` diagnostic is invalid because it transposed the paper image axes
and is excluded from all reported results. The axis-correct one-epoch timing
projects to about 2.66 hours for 400 epochs before one-time load/export overhead.

## Formal Run

Scheduler queue:

```text
queues/paper_indoor_source_paper400_seed2026.json
```

Scheduler job:

```text
jobs/paper_indoor_source_paper400_seed2026.json
```

Output root:

```text
runs/scheduler/paper_indoor_source_paper400_seed2026
```

The task trains 400 epochs, restores `best.pt` for one standalone test pass,
writes unified aggregate metrics, writes `metrics_by_snr.csv`, and retains all
2889 prediction rows.

### Completion

The Scheduler task completed on the first attempt with no monitor warnings:

| Stage | Wall time |
|---|---:|
| 400-epoch train plus per-epoch test | 6051.03 s (1:40:51) |
| Standalone best-checkpoint test | 12.21 s |
| Complete Scheduler task | 6064.28 s (1:41:04) |

Median epoch time was `15.00 s`; the sum of the 400 recorded epoch times was
`6030.73 s`. Scheduler GPU telemetry is device-level and was contaminated by
other processes sharing GPU 0, so its 11.6-GiB device peak is not an MPRI
process-memory measurement. The observed MPRI process allocation during the run
was approximately 1.7 GiB.

The learning-rate trace was verified at every transition:

```text
epochs   1-50: 1.0000e-3
epochs  51-100: 3.0000e-4
epochs 101-150: 9.0000e-5
epochs 151-200: 2.7000e-5
epochs 201-250: 8.1000e-6
epochs 251-300: 2.4300e-6
epochs 301-350: 7.2900e-7
epochs 351-400: 2.1870e-7
```

### Aggregate Test Result

The standalone pass restored the test-RMSE-selected `best.pt`. It was selected
at epoch 400, so `best.pt` and `last.pt` represent the same final model in this
run.

| Mean | MSE | Median | RMSE | CDF@1m | CDF@2m | P90 | P95 | P99 | Count |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.1146 m | 7.2252 m² | 1.6768 m | 2.6880 m | 25.10% | 59.40% | 4.0502 m | 5.1443 m | 8.3124 m | 2889 |

The lowest per-epoch test mean was `2.1072 m` at epoch 357, but the paper-style
selection rule is RMSE, whose minimum was `2.6879 m` at epoch 400. The
independent export differs only by normal floating-point/batch aggregation
rounding.

### Result By Released SNR Variant

| SNR | Mean | Median | RMSE | CDF@1m | CDF@2m | P95 | Count |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 dB | 2.0778 m | 1.6691 m | 2.6314 m | 26.38% | 59.19% | 5.0719 m | 963 |
| 20 dB | 2.0849 m | 1.6511 m | 2.6456 m | 25.23% | 60.02% | 5.0437 m | 963 |
| 10 dB | 2.1811 m | 1.7026 m | 2.7842 m | 23.68% | 58.98% | 5.2874 m | 963 |

All 963 held-out UE groups appear exactly once in every SNR variant. The modest
degradation from 50 dB to 10 dB is directionally consistent with the intended
augmentation, but the absolute localization accuracy is not close to the paper.

### Paper Comparison

| Metric | Paper Scenario 1 | This run |
|---|---:|---:|
| Mean error | 0.28 m | 2.1146 m |
| Fraction at 0.28 m | 66% | 1.73% |
| Fraction at 0.51 m | 90% | 7.06% |

The full original-data chain is operational, but this run does **not**
numerically reproduce the published indoor result. It is about 7.55 times worse
in mean error. The released CFR values, axis conversion, grouped split counts,
model input/output shape, epoch count, LR schedule, and metric export were all
audited. Remaining plausible sources of the gap are details the paper does not
fully specify or release: exact split membership/seed, optimizer and batch
size, training loss, any additional normalization, and exact layer-level
implementation details such as the output reduction. These are hypotheses, not
established causes.

### Artifact Audit

- Training trajectory: 400 data rows, epochs 1 through 400.
- Checkpoints: both record epoch 400, five input channels, 3D output, and
  `normalization: none`.
- Split: 11559 train images, zero validation images, 2889 test images, 4816
  unique UE groups, and no train/test group overlap.
- Predictions: 2889 finite rows and 2889 unique sample keys.
- Test composition: 963 unique held-out positions and 963 rows at each of
  50/20/10 dB.
- Independent metric recomputation from `predictions.csv` matches the exported
  aggregate and per-SNR CSV files.

The compact tracked result table is
`paper_indoor_source_paper400_seed2026_metrics.csv`. Full predictions,
checkpoints, logs, and converted data remain under ignored `runs/` and `data/`
paths.

## Image-Level Random-Split Hypothesis Test

The paper states that the 14448 augmented CFR images are divided 80/20, but it
does not explicitly state that all three SNR variants of one UE remain in the
same split. A controlled A/B therefore keeps every setting above unchanged and
changes only the split unit from UE group to individual augmented image.

Config and runtime files:

```text
config/train/paper_indoor_source_image_random_paper400.yaml
config/eval/paper_indoor_source_image_random_paper400.yaml
queues/paper_indoor_source_image_random_paper400_seed2026.json
jobs/paper_indoor_source_image_random_paper400_seed2026.json
```

The split fractions are written as `11559/14448` and `2889/14448` so that the
task preserves the exact sample counts reported by the paper. With seed 2026,
the declared split has the following properties:

| Audit item | Value |
|---|---:|
| Train images | 11559 |
| Test images | 2889 |
| Unique train positions | 4777 |
| Unique test positions | 2361 |
| Positions represented in both splits | 2322 |
| Test rows whose position also occurs in train | 2772 / 2889 (95.95%) |
| Positions appearing only in test | 39 |

This is intentionally a leakage diagnostic, not the baseline's recommended
unseen-UE evaluation protocol. Its result must be reported beside, and never
silently replace, the UE-grouped result.

### Completion

The Scheduler task completed on the first attempt with no failures, retries, or
monitor warnings:

| Stage | Wall time |
|---|---:|
| 400-epoch train plus per-epoch test | 3395.97 s (56:36) |
| Standalone best-checkpoint test | 15.32 s |
| Complete Scheduler task | 3412.25 s (56:52) |

The run used physical GPU 1 while it was otherwise empty. Scheduler telemetry
recorded a device-level peak of `1727 MiB` and mean utilization of `57.53%`.
The median epoch time was `8.63 s`, and the sum of the 400 recorded epoch times
was `3373.06 s`.

The test-RMSE-selected checkpoint is epoch 305. Its standalone evaluation
contains 2889 finite prediction rows and reproduces the per-epoch metric within
normal floating-point and batch-aggregation rounding.

### Aggregate A/B Result

| Split unit | Mean | MSE | Median | RMSE | CDF@1m | CDF@2m | P95 | Count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UE group (formal) | 2.1146 m | 7.2252 m² | 1.6768 m | 2.6880 m | 25.10% | 59.40% | 5.1443 m | 2889 |
| Augmented image (diagnostic) | 0.5140 m | 0.4519 m² | 0.3926 m | 0.6723 m | 91.10% | 98.86% | 1.2453 m | 2889 |

Changing only the split unit reduces mean error by `75.69%`. This strongly
supports image-level split leakage as a major explanation for the numerical
gap. It does not prove that the authors used this split, because the paper does
not release the split membership, seed, or code.

The image-level result still does not reproduce the reported `0.28 m` mean:
it is `1.84x` larger. Its CDF is `28.38%` at `0.28 m` and `66.08%` at
`0.51 m`, whereas the paper reports `66%` and `90%`, respectively.

### Same-Position Leakage Audit

Because each physical UE has three released SNR images, random image splitting
usually leaves another SNR realization of a test position in the training set.
The observed `95.95%` leakage rate agrees with the theoretical probability
`1 - 0.2^2 = 96%`.

| Same-position train evidence | Mean | MSE | Median | RMSE | CDF@1m | CDF@2m | P95 | Count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| None: truly unseen UE | 1.5685 m | 3.5504 m² | 1.3476 m | 1.8842 m | 27.35% | 80.34% | 3.6757 m | 117 |
| One other SNR image | 0.5560 m | 0.4475 m² | 0.4487 m | 0.6689 m | 89.67% | 99.44% | 1.2605 m | 900 |
| Two other SNR images | 0.4280 m | 0.2604 m² | 0.3547 m | 0.5103 m | 95.78% | 99.73% | 0.9477 m | 1872 |
| Any other SNR image | 0.4695 m | 0.3211 m² | 0.3811 m | 0.5667 m | 93.80% | 99.64% | 1.0874 m | 2772 |

The monotonic improvement from no same-position image, to one, to two is direct
evidence that the image-random aggregate is dominated by position leakage
rather than generalization to unseen UEs.

### Result By Released SNR Variant

| SNR | Mean | Median | RMSE | CDF@1m | CDF@2m | P95 | Count |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 dB | 0.4522 m | 0.3614 m | 0.5948 m | 94.81% | 98.96% | 1.0032 m | 963 |
| 20 dB | 0.4214 m | 0.3331 m | 0.5588 m | 96.09% | 99.16% | 0.8961 m | 947 |
| 10 dB | 0.6644 m | 0.5291 m | 0.8268 m | 82.64% | 98.47% | 1.5400 m | 979 |

Unlike the grouped split, random image splitting does not preserve equal SNR
counts in the test set. The 10-dB variant remains materially harder.

### Conclusion

This controlled experiment confirms that split granularity can explain most,
but not all, of the failed numerical reproduction. It narrows the likely
remaining causes to the paper's unreleased split membership and implementation
choices such as optimizer/batch size, loss definition, normalization, and exact
pooling/output reduction. Those factors require separate one-variable
ablations; they are not established causes here.

The project therefore retains the UE-grouped result as its baseline-correct
unseen-position evaluation. The image-random result is an explicit diagnostic
only. Its compact tracked table is
`paper_indoor_source_image_random_paper400_seed2026_metrics.csv`; full
checkpoints, trajectories, and predictions remain under ignored `runs/`.
