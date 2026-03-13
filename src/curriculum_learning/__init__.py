"""
Curriculum Learning Module.

Single source of truth for all training and evaluation configuration.
"""

from .config import (
    Stage,
    Seeds,
    TrainingConfig,
    CONTINUOUS_CURRICULUM,
    DISCRETE_CURRICULUM,
    CURRICULA,
)

from .env import (
    build_metadrive_config,
    CurriculumEnv,
    InlineEvalEnv,
    HeldOutEvalEnv,
    make_env,
    make_inline_eval_env,
    make_held_out_eval_env,
)

from .callbacks import (
    CurriculumCallback,
    CurriculumEvalCallback,
    EarlyStoppingCallback,
)

from .train import run_training, run_discrete, run_continuous

__all__ = [
    "Stage",
    "Seeds",
    "TrainingConfig",
    "CONTINUOUS_CURRICULUM",
    "DISCRETE_CURRICULUM",
    "CURRICULA",
    "build_metadrive_config",
    "CurriculumEnv",
    "InlineEvalEnv",
    "HeldOutEvalEnv",
    "make_env",
    "make_inline_eval_env",
    "make_held_out_eval_env",
    "CurriculumCallback",
    "CurriculumEvalCallback",
    "EarlyStoppingCallback",
    "run_training",
    "run_discrete",
    "run_continuous",
]
