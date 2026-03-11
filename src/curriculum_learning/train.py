"""
train.py

Central training script. All configuration lives here.
Run with:

    uv run src/train.py discrete
    uv run src/train.py continuous
    uv run src/train.py all        # runs discrete then continuous sequentially
"""

import sys
from curriculum_lib import run_training

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

DEVICE          = "cpu"
N_ENVS          = 4
N_STEPS         = 2048
BATCH_SIZE      = 512
TRAIN_SEED      = 0
EVAL_SEED       = 5000
EVAL_FREQ_STEPS = 100_000    # Evaluate every 100k env steps
N_EVAL_EPISODES = 20   # was 5 — gives 5% resolution on rates

# ---------------------------------------------------------------------------
# Curriculum stages
# ---------------------------------------------------------------------------

DISCRETE_CURRICULUM = [
    {
        "name":                  "Stage 0 — Straight roads, no traffic",
        "map":                   3,
        "num_scenarios":         5,
        "traffic_density":       0.0,
        # Loose reward floor — success/survival rates are what really matter
        "reward_threshold":      225,
        "min_survival_rate":     1.0,
        "min_success_rate":      0.8,   # 80%: one failure in 10 is acceptable
        "discrete_action":       True,
        "discrete_steering_dim": 3,
        "discrete_throttle_dim": 3,
    },
    {
        "name":                  "Stage 1 — Curves added, no traffic",
        "map":                   4,
        "num_scenarios":         10,
        "traffic_density":       0.0,
        "reward_threshold":      200,
        "min_survival_rate":     1.0,
        "min_success_rate":      0.9,
        "discrete_action":       True,
        "discrete_steering_dim": 5,
        "discrete_throttle_dim": 3,
    },
]

# train.py — revised stage definitions
CONTINUOUS_CURRICULUM = [
    {
        "name":              "Stage 1 — Straight roads, no traffic (continuous warm-up)",
        "map":               3,
        "num_scenarios":     20,    # was 5
        "traffic_density":   0.0,
        "reward_threshold":  200,
        "min_survival_rate": 0.95,
        "min_success_rate":  0.85,
        "discrete_action":   False,
        "discrete_steering_dim": None,
        "discrete_throttle_dim": None,
    },
    {
        "name":              "Stage 2 — Curves, no traffic",
        "map":               4,
        "num_scenarios":     50,    # was 10
        "traffic_density":   0.0,
        "reward_threshold":  220,
        "min_survival_rate": 0.90,
        "min_success_rate":  0.80,
        "discrete_action":   False,
        "discrete_steering_dim": None,
        "discrete_throttle_dim": None,
    },
    {
        "name":              "Stage 3 — Mixed roads, light traffic (0.05)",
        "map":               5,
        "num_scenarios":     100,   # was 15
        "traffic_density":   0.05,
        "reward_threshold":  180,
        "min_survival_rate": 0.80,
        "min_success_rate":  0.65,
        "discrete_action":   False,
        "discrete_steering_dim": None,
        "discrete_throttle_dim": None,
    },
    {
        "name":              "Stage 4 — Mixed roads, moderate traffic (0.1)",
        "map":               5,
        "num_scenarios":     100,   # was 15
        "traffic_density":   0.1,
        "reward_threshold":  160,
        "min_survival_rate": 0.75,
        "min_success_rate":  0.60,
        "discrete_action":   False,
        "discrete_steering_dim": None,
        "discrete_throttle_dim": None,
    },
    {
        "name":              "Stage 5 — Random maps, moderate traffic",
        "map":               7,
        "num_scenarios":     200,   # was 20
        "traffic_density":   0.2,
        "reward_threshold":  200,
        "min_survival_rate": 0.75,
        "min_success_rate":  0.70,
        "discrete_action":   False,
        "discrete_steering_dim": None,
        "discrete_throttle_dim": None,
    },
    {
        "name":              "Stage 6 — Random maps, heavy traffic",
        "map":               7,
        "num_scenarios":     500,   # was 100
        "traffic_density":   0.5,
        "reward_threshold":  float("inf"),
        "min_survival_rate": 0.0,
        "min_success_rate":  0.0,
        "discrete_action":   False,
        "discrete_steering_dim": None,
        "discrete_throttle_dim": None,
    },
]

# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_discrete():
    run_training(
        curriculum=DISCRETE_CURRICULUM,
        run_name="discrete",
        n_envs=N_ENVS,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        total_timesteps=50_000_000,
        train_seed=TRAIN_SEED,
        eval_seed=EVAL_SEED,
        device=DEVICE,
        eval_freq_steps=EVAL_FREQ_STEPS,
        n_eval_episodes=N_EVAL_EPISODES,
    )


def run_continuous():
    run_training(
        curriculum=CONTINUOUS_CURRICULUM,
        run_name="continuous",
        n_envs=N_ENVS,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        total_timesteps=50_000_000,
        train_seed=TRAIN_SEED,
        eval_seed=EVAL_SEED,
        device=DEVICE,
        eval_freq_steps=EVAL_FREQ_STEPS,
        n_eval_episodes=N_EVAL_EPISODES,
        # Transfer trunk from discrete phase if available
        trunk_source_ckpt="models/best_discrete/best_model.zip",
        trunk_source_vecnorm="models/best_discrete/vecnorm.pkl",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

PHASES = {
    "discrete":   run_discrete,
    "continuous": run_continuous,
    "all":        lambda: (run_discrete(), run_continuous()),
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in PHASES:
        print(f"Usage: python train.py [{' | '.join(PHASES)}]")
        sys.exit(1)

    PHASES[sys.argv[1]]()