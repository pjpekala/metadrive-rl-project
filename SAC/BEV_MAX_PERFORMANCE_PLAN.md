# BEV Max-Performance Plan (Desktop Safe)

Last updated: 2026-04-14 (implementation started)
Owner: ongoing implementation across sessions

## Goal

Maximize MetaDrive BEV model performance on this desktop system, even if training takes multiple days, while avoiding system crashes (RAM/VRAM OOM and unstable runs).

## System Constraints (Current Desktop)

- GPU: NVIDIA GeForce RTX 3060 Ti (8 GB VRAM)
- CPU: 12 cores
- RAM: 15 GB total, ~12 GB available during idle
- Swap: 4 GB

Practical implication: image-based training must remain conservative (`n_envs=1`, `vec_env=dummy`, no inline eval, controlled replay/batch sizes).

## Current Best Known Starting Point

Use this checkpoint as the resume baseline:

- `models/trained_bev2/bev/checkpoints/sac_metadrive_2600000_steps.zip`

Why: in held-out spot checks, this checkpoint outperformed nearby alternatives and the current `models/bev` lineage.

## Implementation Strategy

1. Patch training resume behavior so resumed checkpoints obey desktop-safe runtime hyperparameters.
2. Resume from the strongest checkpoint (`2600000`) into a new experiment directory.
3. Train in stages with lower learning rates to reduce late-stage collapse.
4. Evaluate frequently and select best checkpoint by held-out metrics (not by latest timestep).
5. Promote only the best evaluated checkpoint to canonical BEV model.

## Required Code Patch

File: `train_metadrive.py`

### Patch Requirements

- Add `--resume-from` CLI argument and pass to `train_sac_agent(...)`.
- Extend `train_sac_agent(...)` signature with `resume_from: Optional[str] = None`.
- When loading, use `resume_source = Path(resume_from) if resume_from else model_path`.
- Load with `SAC.load(..., custom_objects=...)` so these values are overrideable on resume:
  - `learning_rate`
  - `buffer_size`
  - `learning_starts`
  - `batch_size`
  - `train_freq`
  - `gradient_steps`
- For image modalities, force `replay_buffer_class=Uint8ImageDictReplayBuffer` on resume.
- Print effective post-load hyperparameters so we can verify overrides are active.

## Training Profile (Desktop-Safe, High Quality)

Use for BEV continuation:

- `--modality bev`
- `--n-envs 1`
- `--vec-env dummy`
- `--device cuda`
- `--no-inline-eval`
- `--memory-budget-gb 12`
- `--buffer-size 100000`
- `--learning-starts 10000`
- `--batch-size 256`
- `--train-freq 8`
- `--gradient-steps 4`

## Staged Run Plan

Run in a clean destination so old runs are preserved:

- `--save-dir ./models/bev_best_from_2600k`

### Stage 1

- Resume from 2.6M and train to 3.6M total
- Learning rate: `5e-5`

### Stage 2 (if improving)

- Continue to 4.2M total
- Learning rate: `3e-5`

### Early Stop / Freeze Rule

If held-out success rate and reward regress for 2+ evaluation windows, stop and keep the best prior checkpoint.

## Evaluation Protocol

- Fast gate: 50 episodes every 200k-step checkpoint.
- Promotion test: 200 episodes for top 2 checkpoints.
- Final confirmation: 200 episodes on the selected winner.
- Append all results to `evaluation_results.csv`.

## Promotion Rule

- If no new checkpoint beats the 2.6M baseline on held-out evaluation, keep baseline.
- If a new checkpoint wins, promote it as canonical BEV model:
  - target: `models/bev/sac_metadrive_final.zip` (or another agreed canonical path)

## Commands (Reference)

### Stage 1 command

```bash
uv run python train_metadrive.py \
  --modality bev \
  --resume-from models/trained_bev2/bev/checkpoints/sac_metadrive_2600000_steps.zip \
  --save-dir ./models/bev_best_from_2600k \
  --total-timesteps 3600000 \
  --learning-rate 0.00005 \
  --buffer-size 100000 \
  --learning-starts 10000 \
  --batch-size 256 \
  --train-freq 8 \
  --gradient-steps 4 \
  --n-envs 1 \
  --vec-env dummy \
  --device cuda \
  --memory-budget-gb 12 \
  --no-inline-eval
```

### Stage 2 command (only if Stage 1 improves)

```bash
uv run python train_metadrive.py \
  --modality bev \
  --save-dir ./models/bev_best_from_2600k \
  --total-timesteps 4200000 \
  --learning-rate 0.00003 \
  --buffer-size 100000 \
  --learning-starts 10000 \
  --batch-size 256 \
  --train-freq 8 \
  --gradient-steps 4 \
  --n-envs 1 \
  --vec-env dummy \
  --device cuda \
  --memory-budget-gb 12 \
  --no-inline-eval
```

## Todo Management Instructions (Use Every Session)

This section is mandatory process guidance for multi-session implementation.

1. Before starting work each session:
   - Review the todo checklist below.
   - Mark exactly one item as in progress by adding `(IN PROGRESS)`.
2. During implementation:
   - Add new tasks when discovered.
   - Split large tasks into smaller checkbox items.
   - Update task ordering if priorities change.
3. When a task is done:
   - Change `[ ]` to `[x]` immediately.
   - Add a short note in the Session Log (what changed, file paths, key command).
4. If a task becomes irrelevant:
   - Mark it as cancelled using `[x]` and append `(CANCELLED: reason)`.
5. Never leave ambiguous status:
   - Every active task must be either unchecked or checked; avoid stale "in progress" markers.

## Master Todo Checklist

- [x] Patch `train_metadrive.py` to add `--resume-from` and resume source selection.
- [x] Add `custom_objects` override logic for resume-safe hyperparameters.
- [x] Ensure image resume forces `Uint8ImageDictReplayBuffer`.
- [x] Print effective post-load hyperparameters after resume.
- [x] Run lint/tests needed to validate patch safety (`just lint`, targeted test if needed). (Validated via `py_compile` + zero-step resume smoke test; `just lint` recipe not present in current `justfile`.)
- [x] Run 200-episode baseline evaluation for `2600000` checkpoint.
- [x] Launch Stage 1 training (2.6M -> 3.6M) in `./models/bev_best_from_2600k`.
- [x] Evaluate Stage 1 checkpoints every 200k (50-episode quick gate).
- [x] Run 200-episode promotion eval on top Stage 1 candidates.
- [x] Decide go/no-go for Stage 2 based on metrics trend.
- [x] If go: launch Stage 2 training (3.6M -> 4.2M) at LR `3e-5`.
- [ ] Evaluate Stage 2 checkpoints and select best overall model. (IN PROGRESS)
- [ ] Promote winner to canonical BEV path and record final metrics.
- [ ] Update this document with final outcomes and close all todos.

## Session Log

Use one bullet per significant action.

- 2026-04-14: Created this plan file and initialized master todo checklist.
- 2026-04-14: Implemented resume patch in `train_metadrive.py` (`--resume-from`, `--save-dir`, resume source selection, resume-time hyperparameter overrides, and effective-hparams logging).
- 2026-04-14: Validated patch with `uv run python -m py_compile train_metadrive.py` and a zero-step resume smoke test from `models/trained_bev2/bev/checkpoints/sac_metadrive_2600000_steps.zip`.
- 2026-04-14: Completed 200-episode baseline eval for `2600000` checkpoint. Results: success `0.225`, collision `0.510`, out_of_road `0.260`, mean_reward `226.46`.
- 2026-04-14: Started Stage 1 in background with `nohup`; command logs to `models/bev_best_from_2600k/bev/stage1_train.log` and currently resumes correctly from `models/trained_bev2/bev/checkpoints/sac_metadrive_2600000_steps.zip`.
- 2026-04-15: Stage 1 completed to target `3,600,000` timesteps (confirmed from `models/bev_best_from_2600k/bev/sac_metadrive_final.zip` metadata). tmux session ending was expected after command completion.
- 2026-04-15: Completed evaluation sweep for Stage 1 checkpoints and final model (200 episodes each). Results saved to `models/bev_best_from_2600k/bev/stage1_eval_sweep_200.csv` and appended to `evaluation_results.csv`; current best by success is `models/bev_best_from_2600k/bev/checkpoints/sac_metadrive_3411856_steps.zip` (success `0.305`, collision `0.520`, out_of_road `0.185`, mean_reward `234.24`).
- 2026-04-15: Go decision for Stage 2: proceed from best Stage 1 checkpoint `models/bev_best_from_2600k/bev/checkpoints/sac_metadrive_3411856_steps.zip` (not Stage 1 final). Launched Stage 2 in tmux session `bev_stage2` with LR `3e-5`, target `4,200,000` timesteps.
- 2026-04-16: Stage 2 completed to target `4,200,000` timesteps (confirmed from `models/bev_best_from_2600k/bev/sac_metadrive_final.zip` metadata). Completed Stage 2 evaluation sweep (200 episodes each) and saved results to `models/bev_best_from_2600k/bev/stage2_eval_sweep_200.csv`. Best overall checkpoint remains `models/bev_best_from_2600k/bev/checkpoints/sac_metadrive_3411856_steps.zip` (success `0.330`, collision `0.470`, out_of_road `0.180`, mean_reward `244.05`).

## Final Deliverables

- Patched and validated resume-safe trainer.
- Best BEV checkpoint selected by held-out evaluation protocol.
- Canonical BEV model updated.
- `evaluation_results.csv` updated with baseline + final confirmation entries.
- This document fully checked off with concise session history.
