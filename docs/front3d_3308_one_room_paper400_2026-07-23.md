# Front3D 3308 One-Room MPRI Paper400 Diagnostic

## Purpose

This diagnostic tests whether the current MPRI reproduction can achieve a test
mean localization error below 1 m on a geometrically simple Front3D scene.
It is a single-scene capability check, not a collection-level baseline result.

## Scene And Data Contract

- Scene: `front3d_3308`
- Scene class: `small_room`
- Room count: one, verified from `rooms/room_id`
- Room area: approximately 33.4 m2
- UE count: 119
- BS count and contract IDs: four, `[0, 1, 2, 3]`
- m117 UE split: 95 train, 12 validation, 12 test
- Input: all 768 original subcarriers
- Representation: `log_abs`
- Noise data: offline CFR variants at 50, 20, and 10 dB
- Converted sample count: 357, with three SNR variants per UE
- Converted tensor contract: `(W=32, H=768, C=4, N=357)`

The converter applies the m117 link allow-list before grouping links and reads
the four BS IDs from the central `bs_selection_v1` contract.

## Training And Evaluation

- Model: current paper-audited MPRI implementation, 2,628,674 trainable
  parameters for the four-channel 2D task
- Epochs: 400
- Batch size: 16
- Optimizer: Adam, learning rate `1e-3`, weight decay 0
- Schedule: StepLR every 50 epochs with factor 0.3
- Dropout: 0.2
- AMP: disabled
- Checkpoint selection: validation RMSE
- Test use: one independent evaluation of `best.pt` after training
- Scheduler admission: one GPU selected from `[0, 5, 6, 7]`, with at least
  12 GB free memory and a 9 GB reservation. Shared admission is allowed only
  when at most three tasks share the GPU and their declared reservations fit.

The one-epoch CUDA smoke completed without OOM. It observed approximately
7.7 GB peak GPU memory, took 14.0 seconds, and produced validation RMSE
1.941 m. The full paper400 result is pending.

## Runtime Files

- Data config:
  `../config/data/front3d_3308_one_room_multibs_raw768_paper400.yaml`
- Train config:
  `../config/train/front3d_3308_one_room_multibs_raw768_paper400.yaml`
- Eval config:
  `../config/eval/front3d_3308_one_room_multibs_raw768_paper400.yaml`
- Scheduler queue:
  `../queues/front3d_3308_one_room_multibs_raw768_paper400.json`
- Scheduler job:
  `../jobs/front3d_3308_one_room_multibs_raw768_paper400.json`
