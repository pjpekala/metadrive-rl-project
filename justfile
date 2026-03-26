# MetaDrive RL Project - Just commands

# Show available commands
default:
    @just --list

# Train with specified phase: discrete, continuous, or all
train phase:
    uv run -m src.curriculum_learning.train {{phase}}

# Evaluate trained model
# Usage: just eval continuous
#        just eval continuous --stage 2
#        just eval continuous --all-stages
#        just eval continuous --render
eval phase="continuous" *args="":
    uv run -m src.curriculum_learning.evaluate {{phase}} {{args}}

# Open TensorBoard
tensorboard:
    uv run tensorboard --logdir=logs/ --port=6006 

# Clean old models and logs (careful!)
clean:
    rm -rf models/checkpoints_* models/best_*/ models/*/final_*.zip models/*/final_vecnorm.pkl
    rm -rf logs/run_*/

# Clean discrete model weights and logs
clean-discrete:
    rm -rf models/checkpoints_discrete models/best_discrete models/discrete/
    rm -rf logs/run_discrete/

# Clean continuous model weights and logs
clean-continuous:
    rm -rf models/checkpoints_continuous models/best_continuous models/continuous/
    rm -rf logs/run_continuous/

# Quick replay - watch trained agent drive
replay phase="continuous":
    uv run -m src.curriculum_learning.evaluate {{phase}} --render

# Show status of trained models
status:
    @echo "=== Trained Models ==="
    @for dir in models/best_*/; do \
        if [ -f "$$dir/best_model.zip" ]; then \
            echo "✓ $$(basename $$dir)"; \
        fi; \
    done
    @echo ""
    @echo "=== Checkpoints ==="
    @ls -la models/checkpoints_*/ 2>/dev/null | head -20 || echo "No checkpoints found"

# ============== SAC Training ==============

# Train SAC with default LiDAR observations
sac:
    uv run src/sac_training/train.py --n-envs 8 --total-timesteps 1000000

# Train SAC with custom settings
sac-train n timesteps:
    uv run src/sac_training/train.py --n-envs {{n}} --total-timesteps {{timesteps}}

# Train SAC with BEV image observations
sac-bev:
    uv run src/sac_training/train.py --bev --n-envs 8 --total-timesteps 1000000

# Train SAC with BEV custom settings
sac-bev-train n timesteps:
    uv run src/sac_training/train.py --bev --n-envs {{n}} --total-timesteps {{timesteps}}

# Train SAC with live rendering (watch the car drive!)
sac-live:
    uv run src/sac_training/train.py --live-render --n-envs 1 --total-timesteps 100000

# Train SAC with BEV + live rendering
sac-bev-live:
    uv run src/sac_training/train.py --bev --live-render --n-envs 1 --total-timesteps 100000

# Train SAC with video recording
sac-video:
    uv run src/sac_training/train.py --record-video --video-interval 50 --n-envs 4 --total-timesteps 1000000

# Test trained SAC model
sac-test:
    uv run src/sac_training/train.py --test models/sac_metadrive/sac_metadrive_final --test-episodes 10

# Replay SAC agent (watch trained agent drive)
sac-replay:
    uv run src/sac_training/train.py --test models/sac_metadrive/sac_metadrive_final --test-episodes 3 --render

# Clean SAC models and logs
clean-sac:
    rm -rf models/sac_metadrive/
    rm -rf logs/run_sac_metadrive/
