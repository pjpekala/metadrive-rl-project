# Agent Guidelines for MetaDrive RL Project

## Project Overview

This is a deep reinforcement learning project that trains autonomous driving agents using the MetaDrive simulator and stable-baselines3 PPO. The codebase uses curriculum learning to progressively increase environment difficulty.

## Build, Lint, and Test Commands

### Package Management
```bash
# Install dependencies
uv sync

# Add new dependency
uv add <package>

# Add dev dependency
uv add --dev ruff
```

### Linting (ruff)
```bash
# Run linter on all source
uv run ruff check src/

# Run linter with auto-fix
uv run ruff check src/ --fix

# Format code
uv run ruff format src/
```

### Running the Project
```bash
# Train discrete action space model
just train discrete

# Train continuous action space model
just train continuous

# Train both sequentially
just train all

# Evaluate model
just eval continuous
just eval continuous --stage 2
just eval continuous --all-stages
just eval continuous --render

# Watch trained agent
just replay continuous

# Open TensorBoard
just tensorboard

# View model status
just status
```

### Manual Module Execution
```bash
# Training
uv run -m src.curriculum_learning.train discrete
uv run -m src.curriculum_learning.train continuous

# Evaluation
uv run -m src.curriculum_learning.evaluate discrete
uv run -m src.curriculum_learning.evaluate continuous --stage 1
uv run -m src.curriculum_learning.evaluate continuous --render
```

## Code Style Guidelines

### Python Version
- Python 3.11+ (strict requirement per `pyproject.toml`)
- Do not use syntax or libraries unavailable in Python 3.11

### Imports
- Use absolute imports from package root: `from src.curriculum_learning.config import ...`
- Use relative imports within modules: `from .config import ...`
- Group imports: stdlib → third-party → local
- Do not use wildcard imports (`from module import *`)
- Always use `from typing import ...` for type hints

### Type Annotations
- Use type hints for all function parameters and return values
- Use `Optional[X]` for nullable types, not `X | None`
- Use `Union[X, Y]` for union types, not `X | Y` (compatible with Python 3.11)
- Import dataclasses with `from dataclasses import dataclass, field

### Naming Conventions
| Element | Convention | Example |
|---------|------------|---------|
| Modules | snake_case | `vecnorm_utils.py` |
| Classes | PascalCase | `CurriculumCallback` |
| Functions | snake_case | `make_env`, `transfer_trunk` |
| Constants | SCREAMING_SNAKE_CASE | `N_EVAL_EPISODES`, `PROFILES` |
| Variables | snake_case | `train_seed`, `stage_holder` |
| Type variables | PascalCase | (standard Python convention) |

### Dataclasses for Configuration
- Use `@dataclass` for all configuration objects (see `config.py`)
- Use `@property` methods for computed values (e.g., `is_discrete`)
- Use `field(default_factory=...)` for mutable defaults
- Group related fields together logically

### Docstrings
- Use triple-quoted docstrings for modules and public classes/functions
- Use Google-style docstrings with Args/Returns/Raises sections
- Keep docstrings concise but informative

Example:
```python
def make_env(train_seed: int, scenario_index: int, stage_holder: List[Stage]) -> callable:
    """Factory for training environment.
    
    Args:
        train_seed: Base seed for environment randomization
        scenario_index: Worker index for parallel envs
        stage_holder: Shared list reference to current stage
    
    Returns:
        Callable that initializes the environment
    """
```

### Error Handling
- Use specific exception types when possible
- Print informative error messages to stderr
- Include context in error messages (paths, values, expected vs actual)
- For CLI scripts, use `sys.exit(1)` with error messages

Example:
```python
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(
        f"Checkpoint not found at {checkpoint_path}\n"
        f"Run discrete training first with: just train discrete"
    )
```

### Logging and Output
- Use print statements with brackets for module identification: `[ModuleName] message`
- Use TensorBoard logging via `self.logger.record()` in callbacks
- Print progress with meaningful context (timesteps, rates, thresholds)
- Use ASCII characters for status indicators: `✓` success, `✗` failure, `~` warning

### Environment Patterns
- Single source of truth: `build_metadrive_config()` in `env.py`
- Use `stage_holder` list pattern for shared curriculum state
- Use factory functions (`make_env`, `make_inline_eval_env`) for environment creation
- Wrap environments with `Monitor` for logging

### Constants and Magic Numbers
- Define magic numbers as module-level constants (e.g., `N_EVAL_EPISODES = 20`)
- Use descriptive names for thresholds and limits
- Document the meaning of seed ranges in comments

## Code Organization

```
src/curriculum_learning/
├── __init__.py          # Public API exports
├── config.py           # All configuration (Stage, Seeds, TrainingConfig, curricula)
├── env.py              # Environment factories and wrappers
├── train.py            # Training entry point and run_training logic
├── evaluate.py         # Evaluation entry point
├── callbacks.py        # SB3 callback classes
└── vecnorm_utils.py    # VecNormalize utilities
```

### Module Dependencies
- `config.py` - No dependencies on other project modules
- `env.py` - Imports from `config.py`
- `vecnorm_utils.py` - No dependencies on other project modules
- `callbacks.py` - Imports from `vecnorm_utils.py`
- `train.py` - Imports all modules
- `evaluate.py` - Imports from `config.py`, `env.py`, `vecnorm_utils.py`

## Ruff Configuration

The project uses ruff for linting. Key rules enforced:
- F401: Unused imports (always remove)
- F541: f-strings without placeholders (remove `f` prefix)
- E501: Line length (follow existing patterns)

Run `uv run ruff check src/ --fix` before committing to auto-fix issues.

## Important Patterns

### Module Import Guard
For scripts that can run as both modules and direct scripts:
```python
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.curriculum_learning import ...
else:
    from . import ...
```

### VecNormalize Loading Pattern
```python
def load_or_create_vecnorm(path, venv, ...):
    if not os.path.exists(path):
        return VecNormalize(venv, ...)
    try:
        return VecNormalize.load(path, venv)
    except AssertionError as e:
        if "spaces must have the same shape" in str(e):
            return VecNormalize(venv, ...)  # Fresh normalizer
        raise
```

### Stage Progression Pattern
- Use `stage_holder[0]` as a shared reference (list allows mutation)
- On env reset, check if stage changed and update config
- CurriculumCallback advances `stage_holder[0]` when thresholds met

## Common Tasks

### Adding a New Curriculum Stage
Edit `src/curriculum_learning/config.py`:
1. Add new `Stage(...)` to `CONTINUOUS_CURRICULUM` or `DISCRETE_CURRICULUM`
2. Set appropriate thresholds based on difficulty

### Modifying Training Hyperparameters
Edit `TrainingConfig` dataclass in `config.py`:
- `n_envs`: Number of parallel environments
- `n_steps`: Steps per rollout
- `learning_rate`: PPO learning rate
- `net_arch`: Policy/value network architecture

### Adding New Evaluation Metrics
Add to `evaluate_stage()` in `evaluate.py` or `evaluate_policy_verbose()` in `callbacks.py`:
1. Track metric during episode loop
2. Log to TensorBoard with `self.logger.record()`
3. Print summary
