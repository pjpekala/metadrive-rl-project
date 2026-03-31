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

# Available sensor types:
#   lidar       - Default LiDAR (240 lasers), fastest training
#   bev         - Bird's Eye View (5-channel semantic images)
#   rgb         - RGB camera (3-channel front camera)
#   dense-lidar - Dense LiDAR (720 lasers), higher resolution

# Train SAC with default LiDAR observations (fastest)
sac:
    uv run -m src.sac_training.train --sensor-type lidar --n-envs 8 --total-timesteps 1000000

# Train SAC with custom settings
sac-train n timesteps:
    uv run -m src.sac_training.train --sensor-type lidar --n-envs {{n}} --total-timesteps {{timesteps}}

# Train SAC with BEV image observations
sac-bev:
    uv run -m src.sac_training.train --sensor-type bev --n-envs 8 --total-timesteps 1000000

# Train SAC with BEV custom settings
sac-bev-train n timesteps:
    uv run -m src.sac_training.train --sensor-type bev --n-envs {{n}} --total-timesteps {{timesteps}}

# Train SAC with RGB camera observations
sac-rgb:
    uv run -m src.sac_training.train --sensor-type rgb --n-envs 8 --total-timesteps 1000000

# Train SAC with RGB custom settings
sac-rgb-train n timesteps:
    uv run -m src.sac_training.train --sensor-type rgb --n-envs {{n}} --total-timesteps {{timesteps}}

# Train SAC with dense LiDAR observations
sac-dense-lidar:
    uv run -m src.sac_training.train --sensor-type dense-lidar --n-envs 8 --total-timesteps 1000000

# Train SAC with dense LiDAR custom settings
sac-dense-lidar-train n timesteps:
    uv run -m src.sac_training.train --sensor-type dense-lidar --n-envs {{n}} --total-timesteps {{timesteps}}

# Train SAC with live rendering (watch the car drive!)
sac-live:
    uv run -m src.sac_training.train --sensor-type lidar --live-render --n-envs 1 --total-timesteps 100000

# Train SAC with BEV + live rendering
sac-bev-live:
    uv run -m src.sac_training.train --sensor-type bev --live-render --n-envs 1 --total-timesteps 100000

# Train SAC with RGB + live rendering
sac-rgb-live:
    uv run -m src.sac_training.train --sensor-type rgb --live-render --n-envs 1 --total-timesteps 100000

# Train SAC with dense LiDAR + live rendering
sac-dense-lidar-live:
    uv run -m src.sac_training.train --sensor-type dense-lidar --live-render --n-envs 1 --total-timesteps 100000

# Train SAC with video recording
sac-video:
    uv run -m src.sac_training.train --sensor-type lidar --record-video --video-interval 50 --n-envs 4 --total-timesteps 1000000

# Train SAC with BEV + video recording
sac-bev-video:
    uv run -m src.sac_training.train --sensor-type bev --record-video --video-interval 50 --n-envs 4 --total-timesteps 1000000

# Train SAC with RGB + video recording
sac-rgb-video:
    uv run -m src.sac_training.train --sensor-type rgb --record-video --video-interval 50 --n-envs 4 --total-timesteps 1000000

# Train SAC with dense LiDAR + video recording
sac-dense-lidar-video:
    uv run -m src.sac_training.train --sensor-type dense-lidar --record-video --video-interval 50 --n-envs 4 --total-timesteps 1000000

# Test trained SAC model (LiDAR default)
sac-test:
    uv run -m src.sac_training.train --sensor-type lidar --test models/sac_metadrive_lidar/sac_metadrive_final --test-episodes 10

# Test trained SAC model (BEV)
sac-bev-test:
    uv run -m src.sac_training.train --sensor-type bev --test models/sac_metadrive_bev/sac_metadrive_final --test-episodes 10

# Test trained SAC model (RGB)
sac-rgb-test:
    uv run -m src.sac_training.train --sensor-type rgb --test models/sac_metadrive_rgb/sac_metadrive_final --test-episodes 10

# Test trained SAC model (dense LiDAR)
sac-dense-lidar-test:
    uv run -m src.sac_training.train --sensor-type dense-lidar --test models/sac_metadrive_dense-lidar/sac_metadrive_final --test-episodes 10

# Replay SAC agent (LiDAR default)
sac-replay:
    uv run -m src.sac_training.train --sensor-type lidar --test models/sac_metadrive_lidar/sac_metadrive_final --test-episodes 3 --render

# Replay SAC agent (BEV)
sac-bev-replay:
    uv run -m src.sac_training.train --sensor-type bev --test models/sac_metadrive_bev/sac_metadrive_final --test-episodes 3 --render

# Replay SAC agent (RGB)
sac-rgb-replay:
    uv run -m src.sac_training.train --sensor-type rgb --test models/sac_metadrive_rgb/sac_metadrive_final --test-episodes 3 --render

# Replay SAC agent (dense LiDAR)
sac-dense-lidar-replay:
    uv run -m src.sac_training.train --sensor-type dense-lidar --test models/sac_metadrive_dense-lidar/sac_metadrive_final --test-episodes 3 --render

# Clean SAC models and logs (all sensors)
clean-sac:
    rm -rf models/sac_metadrive*/
    rm -rf logs/run_sac_metadrive*/

# Clean SAC models and logs (LiDAR default)
clean-sac-lidar:
    rm -rf models/sac_metadrive_lidar/
    rm -rf logs/run_sac_metadrive_lidar/

# Clean SAC models and logs (BEV)
clean-sac-bev:
    rm -rf models/sac_metadrive_bev/
    rm -rf logs/run_sac_metadrive_bev/

# Clean SAC models and logs (RGB)
clean-sac-rgb:
    rm -rf models/sac_metadrive_rgb/
    rm -rf logs/run_sac_metadrive_rgb/

# Clean SAC models and logs (dense LiDAR)
clean-sac-dense-lidar:
    rm -rf models/sac_metadrive_dense-lidar/
    rm -rf logs/run_sac_metadrive_dense-lidar/
