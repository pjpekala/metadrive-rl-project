"""
Evaluation wrapper.

Delegates to curriculum_learning/evaluate.py for standalone evaluation.

Usage:
    uv run src/evaluate.py                          # continuous, final stage
    uv run src/evaluate.py discrete                 # discrete model
    uv run src/evaluate.py continuous --stage 2     # specific stage
    uv run src/evaluate.py continuous --all-stages # sweep all stages
    uv run src/evaluate.py continuous --render      # watch it drive
"""

from curriculum_learning.evaluate import main

if __name__ == "__main__":
    main()
