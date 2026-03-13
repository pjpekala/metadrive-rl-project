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
    uv run tensorboard --logdir=logs/

# Clean old models and logs (careful!)
clean:
    rm -rf models/checkpoints_* models/best_*/ models/*/final_*.zip models/*/final_vecnorm.pkl
    rm -rf logs/run_*/

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
