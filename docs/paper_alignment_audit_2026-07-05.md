# MPRI Paper Alignment Audit

Date: 2026-07-05

This is an audit note for status tracking.

| Paper item | Current implementation | Status |
|---|---|---|
| CFR image input from flattened MIMO CFR observations | `scripts/convert_cfr_to_csi.py` writes `features (W,H,C,N)` and `labels/position (N,2)`; Front3D stacks links by UE and BS. | Partial: Front3D smoke verified. |
| Noise augmentation | Config supports SNR `[50,20,10]`; AWGN is applied to available CFR/path-derived CFR. | Adaptation: paper adds noise to PHY `Filter_multipath`, unavailable in current datasets. |
| Training schedule | `config/train/front3d_5431_paper400.yaml` uses 400 epochs, lr 1e-3, StepLR step 50 gamma 0.3. | Aligned with located paper text. |
| Train/test split | Paper text uses 0.8 train/test. MPRI config supports train 0.8, val 0.0, test 0.2. | Aligned for paper-style config. |
| Network structure | `src/model.py` implements input Conv-BN-ReLU-Pool, upper multi-path block, lower dual-path residual blocks and output head. | Needs final layer-by-layer review against Fig. 4. |
| Optimizer and batch size | Current paper text search did not locate explicit values. Smoke/paper configs use Adam and batch size 32 for paper-style run. | Requires review/justification. |
| DeepMIMO support | Path parameters are converted to virtual CFR from `power/phase/delay` before MPRI image construction. | Requires user review; DeepMIMO files are not direct CFR. |

Smoke evidence:

| Data | Run | Test mean / median / P95 |
|---|---|---|
| Front3D `front3d_5431` | `runs/eval/mpri_front3d_5431_smoke_eval-2026-07-05-16-03-48` | `2.925 / 2.728 / 3.261 m` |
| DeepMIMO O1_3p5 | `runs/eval/mpri_deepmimo_o1_3p5_smoke_eval-2026-07-05-16-03-48` | `363.769 / 363.417 / 368.848 m` |
