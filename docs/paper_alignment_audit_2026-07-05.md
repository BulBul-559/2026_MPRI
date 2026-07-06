# MPRI Paper Alignment Audit

Date: 2026-07-05

Scope: audit the current MPRI implementation against the paper design and training strategy. This is a status/audit note, not an execution guide.

Evidence reviewed:

- Paper PDF pages around Fig. 3 and Fig. 4, rendered locally to inspect the CFR-image augmentation and network diagram.
- Extracted paper text for Sections III-D, IV-B/IV-C/IV-D, V-A, V-B, V-E and V-G.
- Current implementation: `scripts/convert_cfr_to_csi.py`, `src/model.py`, `src/data.py`, `train_mpri.py`, and `config/train/front3d_5431_paper400.yaml`.

## Summary

| Area | Conclusion |
|---|---|
| Network scale | Aligned at parameter-count level: `MPRINet(in_channels=1, output_dim=3)` has `2,622,019` trainable parameters, matching the paper's approximately 2.6M claim. |
| MPRI module | Aligned in structure: input module, upper half, lower half, residual/cross connections, and output head are implemented. The code wraps the two paper paths inside one `DualPathBasicUnit` per stacked layer. |
| Training schedule | Aligned for paper-style config: 400 epochs, learning rate `1e-3`, StepLR `50/0.3`, and per-epoch test evaluation are configured. |
| Noise augmentation | Partially aligned: SNR levels and augmentation multiplicity match the paper, but current data lacks PHY `Filter_multipath`, so AWGN is applied to offline CFR. |
| Dataset/split | Adapted: the paper uses its two generated scenarios and 3D labels; current Front3D uses project-level UE split and 2D `(x, y)` labels. |
| Optimizer/batch | Paper text inspected here did not specify them. Current paper-style config uses Adam, batch size 32, weight decay 0.0; this remains a documented implementation choice. |
| DeepMIMO | Adapted through shared offline CFR: DeepMIMO v3/v4 production is owned by `DeepMIMOFactory`; MPRI only reads the produced `deepmimo_cfr_v1` HDF5 with `cfr` axes `[link, rx_ant, tx_ant, subcarrier]` and applies the same conversion/noise path as Front3D. |

## Paper-To-Code Mapping

| Paper component | Current code/config | Status |
|---|---|---|
| CFR image as `Ns x Np x Ng` tensor, where `Ng` is BS depth | Converter writes `features (W,H,C,N)` and `CFRDataset` reads it as `(N,C,H,W)`. `H` is selected subcarriers (`Ns`, usually 193), `W` is flattened per-link/channel feature width (`Np` or representation-expanded width), and `C` is selected BS depth (`Ng`). | Aligned with storage-layout adaptation. |
| Frequency down-sampling to `Ns=193` in Fig. 3 | `config/data/front3d_5431_paper.yaml` uses `subcarriers.count: 193`; converter uses uniform subcarrier selection. | Aligned. |
| MIMO CFR flattening | `complex_to_feature()` flattens non-subcarrier CFR axes before writing the CFR image. Front3D stacks links by UE and selected BS. | Aligned at input contract level; exact physical antenna meaning depends on dataset. |
| Noise augmentation at SNR 50/20/10 dB, multiplying samples by `s` | Formal config uses `augmentation.enabled: true`, `snr_db: [50,20,10]`, `include_clean: false`; tests assert the three SNR levels. | SNR/multiplicity aligned; insertion point adapted. |
| Noise insertion before channel-estimation output | Paper adds noise at PHY `Filter_multipath`. Current converter applies complex AWGN to offline CFR because raw `Filter_multipath` is unavailable. | Adaptation; must be described in final reports. |
| Input module: image input plus Conv-BN-ReLU-Pooling | `InputModule` implements Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d. | Aligned; pool kernel is an implementation choice because the paper does not specify dimensions. |
| Upper half: bottleneck, four paths, depth concatenation, identity mapping | `UpperHalf` uses 1x1 bottleneck; paths output 64, 32, 32, 128 channels; path 2 uses 5x1 + 1x3; path 3 uses max-pool + 3x3; identity add after concat. | Aligned. |
| Lower half: two parallel propagation paths, first path 3x3 filters, second path 1x1 filters, cross-connections, stacked three times | `LowerHalf` has three `DualPathBasicUnit` stages. Each unit contains both paper paths and exchanges both branch outputs through addition. Stage 2/3 include MaxPool down-sampling. | Aligned in module semantics. |
| Lower-half addition instead of depth concatenation | `DualPathBasicUnit` and final `LowerHalf` merge paths through addition. | Aligned. |
| Output module: combine two MPRI paths, basic conv block, dropout, final fully connected layer | `OutputModule` uses Conv-BN-ReLU, AdaptiveMaxPool, Dropout, Linear. The pooling step is treated as part of the basic conv/head reduction. | Mostly aligned; exact pooling form is implementation detail. |
| 3D UE output in paper | `MPRINet` supports `output_dim=3`; current Front3D configs use `output_dim=2` because target labels are 2D. | Dataset adaptation. |
| Paper reports MPRI module at about 2.6M parameters | Test added for `in_channels=1`, `output_dim=3`; observed trainable parameters: `2,622,019`. | Aligned. |
| Paper says MPRI has 63 layers | Current code matches parameter count and diagram semantics, but the exact layer-count convention is not reproducible from the prose alone. | Documented review note; not a blocking implementation mismatch. |

## Training Strategy Mapping

| Paper item | Current code/config | Status |
|---|---|---|
| 400 training epochs | `config/train/front3d_5431_paper400.yaml` sets `epochs: 400`. | Aligned. |
| Initial learning rate `1e-3` | Paper-style config sets `optimizer.learning_rate: 0.001`. | Aligned. |
| Periodic learning-rate drop `50/0.3` | Paper-style config uses `StepLR(step_size=50, gamma=0.3)`. | Aligned. |
| Test set evaluated each epoch | Paper-style config sets `evaluate_test_each_epoch: true`. With `val_split=0`, `train_mpri.py` selects the best checkpoint by test RMSE. | Aligned with paper experimental protocol. |
| 0.8 train/test split | Paper-style config uses train/val/test `0.80/0.00/0.20`; `ue_level` grouping avoids duplicated augmented samples crossing split boundaries. | Aligned with leakage-safe adaptation. |
| RMSE and MeanErr metrics | `train_mpri.py` and `evaluate_mpri.py` compute RMSE and mean Euclidean error. | Aligned. |
| Optimizer and batch size | Current paper text search did not locate explicit values; formal config uses Adam, batch size 32, weight decay 0.0. | Documented implementation choice. |
| Gradient clipping, AMP, weight initialization | Code supports clipping and AMP; paper-style config disables AMP. Initialization/clipping are implementation details not found in paper text. | Non-paper extras; keep documented. |

## Remaining Baseline Work

| Item | Tracking |
|---|---|
| Run Front3D collection-level 400-epoch MPRI experiments and report scene/collection metrics | `FEAT-MPRI-001`; single-scene offline-noise evidence exists, collection-level formal test remains. |
| Decide whether to add strongest-RSS single-BS contrast in addition to central-BS default | User review under MPRI row in central checklist |
| Keep DeepMIMOFactory output contract stable for MPRI consumption | `STR-DATA-004` |
| Accept Adam/batch-size/log-abs/2D-output choices as paper adaptations for current Front3D | User review under MPRI row in central checklist |

## Smoke Evidence

| Data | Run | Test mean / median / P95 / RMSE when available |
|---|---|---|
| Front3D `front3d_5431` | `runs/eval/mpri_front3d_5431_smoke_eval-2026-07-05-16-03-48` | `2.925 / 2.728 / 3.261 m` |
| DeepMIMO O1_3p5 legacy path-derived smoke | `runs/eval/mpri_deepmimo_o1_3p5_smoke_eval-2026-07-05-16-03-48` | `363.769 / 363.417 / 368.848 m` |
| DeepMIMO O1_3p5 Factory axis-corrected smoke | `runs/eval/mpri_deepmimo_o1_3p5_smoke_eval-2026-07-06-13-49-47` | `382.858 / 382.845 / 382.895 / 382.858 m` |

Current DeepMIMO input path has since been updated to consume offline CFR HDF5 from `DeepMIMOFactory`; the legacy path-derived smoke row is historical evidence only. The current smoke uses `cfr.shape=(16,16,2,768)`, selects 64 subcarriers, and applies the three SNR levels.

## Offline-Noisy Single-Scene Evidence

| Data | Config | Best checkpoint test mean / median / P95 / RMSE | Last checkpoint test mean / median / P95 / RMSE | Time |
|---|---|---|---|---|
| Front3D `front3d_0015`, noisy 50/20/10 dB | 193 subcarriers, 400 epochs | `0.572 / 0.480 / 1.279 / 0.690 m` at epoch 25 | `0.577 / 0.425 / 1.573 / 0.817 m` at epoch 400 | convert `0:30.37`, train `6:47.28` |
| Front3D `front3d_0015`, noisy 50/20/10 dB | raw 768 subcarriers, 400 epochs | `0.558 / 0.458 / 1.333 / 0.687 m` at epoch 26 | `0.603 / 0.449 / 1.522 / 0.812 m` at epoch 400 | convert `1:01.03`, train `21:52.69` |

Note: with `val_split=0`, `best.pt` is selected by test RMSE to mirror the paper-style per-epoch test evaluation. Use `last.pt` for a non-test-selected endpoint.
