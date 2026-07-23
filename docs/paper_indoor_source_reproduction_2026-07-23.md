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
2889 prediction rows. Final timing and results will replace this pending note
after the Scheduler task completes.
