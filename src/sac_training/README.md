# SAC Training for MetaDrive

This module contains a standalone Soft Actor-Critic (SAC) training implementation for MetaDrive, ported from the original `metadrive_training.py` script.

It features:
- Live rendering during training
- Automatic video recording
- Real-time training visualization
- TensorBoard support

## Usage

Run the training script using `uv run` from the project root:

```bash
# Basic training (no visualization)
uv run src/sac_training/train.py

# Live rendering (watch the car drive - recommended with n-envs=1)
uv run src/sac_training/train.py --live-render --n-envs 1

# Training with video recording
uv run src/sac_training/train.py --record-video --video-interval 50

# Full visualization suite (live render + plots + videos)
uv run src/sac_training/train.py --live-render --visualize --record-video --n-envs 1
```

## Arguments

- `--total-timesteps`: Total training steps (default: 1,000,000)
- `--n-envs`: Number of parallel environments (default: 8)
- `--live-render`: Enable live rendering window
- `--record-video`: Enable periodic video recording
- `--video-interval`: Frequency of video recording in episodes
- `--visualize`: Enable live matplotlib plots
- `--device`: Compute device (cuda/cpu)
- `--test`: Path to a model file to test
- `--test-episodes`: Number of episodes to run during testing

## Outputs

All outputs are saved to `models/sac_metadrive/`:
- `checkpoints/`: Model checkpoints
- `logs/`: TensorBoard logs
- `videos/`: Recorded videos
- `sac_metadrive_final.zip`: Final trained model
