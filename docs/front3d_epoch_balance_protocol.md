# Front3D MPRI Epoch-Balance Protocol

This note records the analysis protocol used for the 2026-07-07 single-BS
MPRI epoch-balance run and the follow-up multi-BS run. The goal is to choose a
practical training budget without selecting checkpoints from target test
metrics.

## Scope

- Dataset: `Front3d_2988_CFR_Only/noise_sub_test_set`
- Split protocol: `m117_single_scene_ue_level`
- Task unit: one independent target scene from the test-scene pool
- Model: MPRI
- Input modalities: CFR fingerprint only
- Checkpoint selection: `best.pt` by validation RMSE
- Formal evaluation: one final inference on the scene `test` split

The epoch-balance experiment is an exploratory training-budget study. It may
use a longer epoch ceiling than the formal reporting budget so that convergence
can be measured.

## Scene Set

Use the same 10 scenes for all compared branches so per-scene difficulty is
paired:

```text
front3d_0015
front3d_0055
front3d_0080
front3d_0095
front3d_0119
front3d_0149
front3d_0179
front3d_0207
front3d_5583
front3d_5704
```

This set intentionally covers small, medium, and large scene tasks in the
current test-scene pool.

## Compared Branches

For the single-BS study:

- `paper193`: 193 uniformly sampled subcarriers
- `raw768`: raw 768 subcarriers with `subcarrier_policy: first`

For the multi-BS study, keep the same two subcarrier branches and change only
the BS view:

- `bs_count: 4`
- BS IDs from the canonical `bs_selection_v1` contract

The selected BS set is fixed per scene by the contract. The padded converted
HDF5 is used so UE samples with partial selected-BS visibility remain
represented with explicit valid-BS metadata.

## Formal Single-BS Contract

The formal Front3D MPRI single-BS setting is:

- Main configuration: `raw768`
- Subcarrier policy: raw 768 subcarriers with `subcarrier_policy: first`
- BS setting: per-scene `single_bs_ids` from
  `baseline/contracts/front3d/Front3d_2988_CFR_Only/m117_single_scene_ue_level/bs_selection_v1/bs_selection_contract.json`
- Training budget: `60` epochs
- Checkpoint for evaluation: validation-RMSE `best.pt`
- Evaluation: one final inference on the scene `test` split

This 60-epoch setting is the current formal convention. Archived 400-epoch
single-BS convergence studies are retained as historical evidence only and do
not override the formal 60-epoch contract.

## Formal Multi-BS Contract

The formal Front3D MPRI multi-BS setting is:

- Main configuration: `raw768`
- Subcarrier policy: raw 768 subcarriers with `subcarrier_policy: first`
- BS setting: per-scene `multi_bs_ids` from
  `baseline/contracts/front3d/Front3d_2988_CFR_Only/m117_single_scene_ue_level/bs_selection_v1/bs_selection_contract.json`
- Multi-BS sample mode: padded
- Training budget: `100` epochs
- Checkpoint for evaluation: validation-RMSE `best.pt`
- Evaluation: one final inference on the scene `test` split

The `paper193` branch is retained as an ablation and compatibility reference,
not the main formal multi-BS configuration.

This 100-epoch setting is the current formal convention. Archived 400-epoch
multi-BS convergence studies are retained as historical evidence only and do
not override the formal 100-epoch contract.

## Training Ceiling

For epoch-balance studies, run each branch with a long ceiling, normally 400 epochs, using the same
optimizer, scheduler, seed, batch size, split plan, and device policy as the
formal MPRI wrapper. This ceiling is for convergence measurement, not a
recommendation by itself. The current formal budgets are fixed by the contracts
above: 60 epochs for single-BS and 100 epochs for multi-BS.

For each run, keep `logging.evaluate_test_each_epoch: false`; the analysis
uses validation metrics only.

## Primary Criterion

Use validation RMSE from each run's training `metrics.csv`.

Definitions:

- Exact best epoch: epoch with minimum `val_rmse`.
- Running best at epoch `e`: best validation RMSE seen from epoch 1 through
  `e`.
- Near-best tolerance: `max(0.05 m, 2% of exact best validation RMSE)`.
- Near-best epoch: earliest epoch where the running best is within that
  tolerance of the 400-epoch exact best.
- Stable checkpoint capture point: maximum near-best epoch across all runs in
  the evidence set.
- Recommended budget: stable checkpoint capture point plus a 20-epoch margin.

The 20-epoch margin is intentionally conservative. It absorbs small scene-level
variance without forcing every formal run to chase rare exact-best epochs.

## Secondary Diagnostics

Compute these only as supporting evidence:

- Exact best epoch distribution.
- Plateau epoch based on smoothed validation-RMSE slope.
- Candidate budget sensitivity for budgets such as 20, 30, 40, 50, 60, 65,
  70, 100, and 400.

Budget sensitivity should report:

- near-best captured count
- exact-best captured count
- mean running-best delta to final best
- p90 running-best delta
- max running-best delta

## Reporting Artifacts

For each study, write an analysis directory under:

```text
baseline/runs/formal/front3d/Front3d_2988_CFR_Only/m117_single_scene_ue_level/aggregate/mpri/convergence/<analysis_id>/
```

Expected files:

- `training_curve_long.csv`: one row per run and epoch.
- `scene_convergence_summary.csv`: one row per scene and branch.
- `budget_checkpoint_sensitivity.csv`: candidate budget table.
- `overall_summary.json`: machine-readable headline summary.
- `convergence_report.md`: human-readable recommendation and key tables.

## Decision Rule

The current formal MPRI epoch convention is unique:

- Single-BS: `60` epochs.
- Multi-BS: `100` epochs.

Do not use test-set metrics to choose or revise these budgets. Test metrics are
only for final formal evaluation after the training budget is fixed.

The 2026-07-07 multi-BS evidence set found a recommended budget of 102 epochs by
the near-best-plus-margin rule; the active formal contract rounds this to 100
epochs. The archived single-BS convergence study suggested 65 epochs, but the
active formal contract intentionally keeps single-BS at 60 epochs for continuity
with the current formal queue and reporting convention. If future evidence
changes materially, record the new budget with its evidence set and sensitivity
table before using it for formal reporting.
