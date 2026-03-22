# MetaDrive RL Project - Just commands

# Show available commands
default:
    @just --list

# Train with specified phase: discrete, continuous, or all
train phase:
    uv run -m src.curriculum_learning.train {{phase}}

# Train BEV sensor experiment
train-bev:
    uv run -m src.curriculum_learning.train_bev

# Train RGB camera sensor experiment
train-rgb:
    uv run -m src.curriculum_learning.train_rgb

# Train dense LiDAR sensor experiment
train-dense-lidar:
    uv run -m src.curriculum_learning.train_dense_lidar

# Evaluate trained model
# Usage: just eval continuous
#        just eval continuous --stage 2
#        just eval continuous --all-stages
#        just eval continuous --render
eval phase="continuous" *args="":
    PYTHONUNBUFFERED=1 uv run -m src.curriculum_learning.evaluate {{phase}} {{args}}

# Evaluate BEV sensor experiment
eval-bev *args="":
    just eval bev {{args}}

# Evaluate RGB camera sensor experiment
eval-rgb *args="":
    just eval rgb {{args}}

# Evaluate dense LiDAR sensor experiment
eval-dense-lidar *args="":
    just eval dense_lidar {{args}}

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
