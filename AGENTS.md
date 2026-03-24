# AGENTS.md - Agentic Coding Guidelines

This file provides guidelines for agents operating in this repository.

## Project Overview

This is a MetaDrive reinforcement learning project that trains autonomous driving agents using curriculum learning with PPO (Proximal Policy Optimization) from stable-baselines3.

## Build / Run Commands

### Installation
```bash
uv sync           # Install all dependencies (including dev: ruff)
```

### Training
```bash
just train discrete    # Train with discrete action space
just train continuous  # Train with continuous action space
just train all         # Train discrete then continuous sequentially
```

### Evaluation
```bash
just eval continuous              # Evaluate final model
just eval continuous --stage 2    # Evaluate specific curriculum stage (1-based)
just eval continuous --all-stages # Evaluate all stages
just eval continuous --render     # Watch the agent drive
just replay continuous             # Shortcut for --render
```

### Monitoring
```bash
just tensorboard  # Open TensorBoard at localhost:6006
```

### Linting & Formatting
```bash
ruff check .      # Lint all Python files
ruff format .     # Format all Python files
```

### Cleaning
```bash
just clean            # Remove all models and logs
just clean-discrete   # Remove discrete model only
just clean-continuous # Remove continuous model only
just status           # Show trained model status
```

### Single Test Execution
No formal test framework exists. Test logic manually if needed.

## Code Style Guidelines

### Imports
Organize imports in three sections (separate by blank line):
1. Standard library
2. Third-party packages
3. Local application imports

Example:
```python
import os
import sys
from typing import Optional, List

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from .config import CONTINUOUS_CURRICULUM, DISCRETE_CURRICULUM, TrainingConfig
```

### Formatting
- Use **ruff** for formatting: `ruff format .`
- Maximum line length: 88 characters (ruff default)
- Use trailing commas in multi-line calls

### Type Hints
- Always use type hints for function signatures
- Use `Optional[T]` instead of `Union[T, None]`
- Use built-in types as hints (not from typing module): `list`, `dict`, `set` (Python 3.11+)

```python
def make_env(
    train_seed: int,
    scenario_index: int,
    stage_holder: List[Stage],
) -> callable:
```

### Naming Conventions
- **Functions/variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `SCREAMING_SNAKE_CASE`
- **Private methods**: prefix with `_`

### Dataclasses for Configuration
Use `@dataclass` for configuration objects (see `config.py`):

```python
@dataclass
class Stage:
    name: str
    map: Union[int, str]
    num_scenarios_train: int
    # ... use field() for mutable defaults
```

### Error Handling
Wrap environment steps with try/except and treat errors as termination:

```python
def step(self, action):
    try:
        return super().step(action)
    except Exception as e:
        print(f"[CurriculumEnv] step() error: {e} — treating as termination")
        obs = super().reset()[0]
        return obs, -10.0, True, False, {}
```

### Docstrings
Use docstrings for public APIs. Follow Google style:

```python
def build_metadrive_config(
    stage: Stage,
    seed: int,
    scenario_index: int = 0,
) -> dict:
    """
    Build a MetaDrive config dict for a given curriculum stage.
    
    This is the SINGLE SOURCE OF TRUTH for environment configuration.
    
    Args:
        stage: Curriculum stage configuration
        seed: Base seed for this environment
        scenario_index: Worker index; offsets seed for different scenarios per worker
    """
```

### Project Structure
```
src/curriculum_learning/
├── __init__.py       # Package marker
├── config.py         # Configuration dataclasses and curriculum definitions
├── env.py            # Environment factories and wrappers
├── train.py          # Main training script
├── evaluate.py       # Evaluation script
└── callbacks.py      # Training callbacks for curriculum learning
```

### MetaDrive Integration
- Environment config goes through `build_metadrive_config()` as single source of truth
- Use `SafeMetaDriveEnv` as base class
- Wrap environments with `Monitor` for tracking

### Curriculum Pattern
- Define stages in `config.py` using `Stage` dataclass
- Use `stage_holder: List[Stage]` pattern for live updates across workers
- Advance curriculum based on success_rate, survival_rate thresholds

### TensorBoard Logging
- Logs stored in `logs/run_{run_name}/`
- Use `tensorboard_log` parameter in PPO
- Record custom metrics via logger.record()

### Model Checkpoints
- Best model: `models/best_{phase}/best_model.zip`
- Final model: `models/{phase}/final_model.zip`
- Weight transfer pattern: load trunk weights from discrete to continuous training

### Key Configuration
All hyperparameters defined in `TrainingConfig` dataclass:
- `n_envs`: 8 parallel environments
- `batch_size`: 512
- `learning_rate`: 1e-4
- `device`: "cpu" (default)
- `net_arch`: dict(pi=[512, 256], vf=[512, 256])

### Common Patterns

**Environment factory pattern:**
```python
def make_env(train_seed: int, scenario_index: int, stage_holder: List[Stage]) -> callable:
    def _init():
        env = CurriculumEnv(...)
        env = Monitor(env)
        return env
    return _init
```

**Vectorized environments:**
```python
train_env = SubprocVecEnv([make_env(...) for i in range(n_envs)])
eval_env = DummyVecEnv([make_eval_env(...)])
```

### Ruff Configuration
The project uses ruff for linting with these rules (add to pyproject.toml if needed):
- E: pycodestyle errors
- F: pyflakes
- I: isort
- N: pep8-naming
- W: pycodestyle warnings