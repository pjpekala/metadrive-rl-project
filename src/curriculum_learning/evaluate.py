"""
evaluate.py

Standalone evaluation script. Loads a trained model and runs it against
a held-out scenario set, printing per-episode stats and a summary.

Usage:
    uv run src/evaluate.py                              # evaluate continuous (default)
    uv run src/evaluate.py discrete                     # evaluate discrete model
    uv run src/evaluate.py continuous                   # evaluate continuous model
    uv run src/evaluate.py continuous --render          # watch it drive
    uv run src/evaluate.py continuous --stage 2         # evaluate on a specific stage
    uv run src/evaluate.py continuous --all-stages      # sweep every stage

Key fix: env config here exactly mirrors curriculum_lib._build_config so that
observations are on the same scale the model was trained on, and termination
conditions match (out_of_road_done=True, crash_vehicle_done=True).
"""

import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv

# ---------------------------------------------------------------------------
# Stage definitions — mirrors train.py curriculum exactly
# ---------------------------------------------------------------------------

DISCRETE_STAGES = [
    {
        "name":                  "Stage 0 — Straight roads, no traffic",
        "map":                   3,
        "num_scenarios":         5,
        "traffic_density":       0.0,
        "discrete_action":       True,
        "discrete_steering_dim": 3,
        "discrete_throttle_dim": 3,
    },
    {
        "name":                  "Stage 1 — Curves added, no traffic",
        "map":                   4,
        "num_scenarios":         10,
        "traffic_density":       0.0,
        "discrete_action":       True,
        "discrete_steering_dim": 5,
        "discrete_throttle_dim": 3,
    },
]

CONTINUOUS_STAGES = [
    {
        "name":              "Stage 1 — Straight roads, no traffic (continuous warm-up)",
        "map":               3,
        "num_scenarios":     5,
        "traffic_density":   0.0,
        "discrete_action":   False,
        "discrete_steering_dim": None,
        "discrete_throttle_dim": None,
    },
    {
        "name":              "Stage 2 — Curves, no traffic",
        "map":               4,
        "num_scenarios":     10,
        "traffic_density":   0.0,
        "discrete_action":   False,
        "discrete_steering_dim": None,
        "discrete_throttle_dim": None,
    },
    {
        "name":              "Stage 3 — Mixed roads, light traffic (0.05)",
        "map":               5,
        "num_scenarios":     15,
        "traffic_density":   0.05,
        "discrete_action":   False,
        "discrete_steering_dim": None,
        "discrete_throttle_dim": None,
    },
    {
        "name":              "Stage 4 — Mixed roads, moderate traffic (0.1)",
        "map":               5,
        "num_scenarios":     15,
        "traffic_density":   0.1,
        "discrete_action":   False,
        "discrete_steering_dim": None,
        "discrete_throttle_dim": None,
    },
    {
        "name":              "Stage 5 — Random maps, moderate traffic",
        "map":               7,
        "num_scenarios":     20,
        "traffic_density":   0.2,
        "discrete_action":   False,
        "discrete_steering_dim": None,
        "discrete_throttle_dim": None,
    },
    {
        "name":              "Stage 6 — Random maps, heavy traffic",
        "map":               7,
        "num_scenarios":     100,
        "traffic_density":   0.5,
        "discrete_action":   False,
        "discrete_steering_dim": None,
        "discrete_throttle_dim": None,
    },
]

PROFILES = {
    "discrete": {
        "model_path":        "models/best_discrete/best_model.zip",
        "vecnorm_path":      "models/best_discrete/vecnorm.pkl",
        "stages":            DISCRETE_STAGES,
        "default_stage_idx": len(DISCRETE_STAGES) - 1,
    },
    "continuous": {
        "model_path":        "models/best_continuous/best_model.zip",
        "vecnorm_path":      "models/best_continuous/vecnorm.pkl",
        "stages":            CONTINUOUS_STAGES,
        "default_stage_idx": len(CONTINUOUS_STAGES) - 1,
    },
}

# Held-out seed — must not overlap with train (0) or inline eval (5000)
EVAL_SEED       = 2000
N_EVAL_EPISODES = 20

# ---------------------------------------------------------------------------
# Env factory — mirrors curriculum_lib._build_config exactly
# ---------------------------------------------------------------------------

def make_eval_env(seed, stage, use_render=False):
    def _init():
        config = {
            "num_scenarios":         stage["num_scenarios"],
            "start_seed":            seed,
            "map":                   stage["map"],
            "traffic_density":       stage["traffic_density"],
            # Reward shaping — must match training exactly
            "driving_reward":        1.0,
            "speed_reward":          0.1,
            "use_lateral_reward":    True,
            "success_reward":        10.0,
            "out_of_road_penalty":   5.0,
            "crash_vehicle_penalty": 5.0,
            # Termination conditions — must match training exactly
            # Without these, off-road episodes run to the full 1000-step horizon
            # accumulating massive negative reward instead of terminating
            "out_of_road_done":      True,
            "crash_vehicle_done":    True,
            "crash_object_done":     True,
            "horizon":               1000,
            "use_render":            use_render,
            "discrete_action":       stage["discrete_action"],
        }
        if stage.get("discrete_action"):
            config["discrete_steering_dim"] = stage["discrete_steering_dim"]
            config["discrete_throttle_dim"] = stage["discrete_throttle_dim"]

        env = SafeMetaDriveEnv(config=config)

        _original_reset = env.reset
        def _reset_no_seed(**kwargs):
            kwargs.pop("seed", None)
            kwargs.pop("options", None)
            return _original_reset(**kwargs)
        env.reset = _reset_no_seed

        env = Monitor(env)
        return env
    return _init


# ---------------------------------------------------------------------------
# Single-stage evaluation
# ---------------------------------------------------------------------------

def evaluate_stage(model, eval_env, n_episodes):
    episode_rewards, episode_lengths = [], []
    success_count = collision_count = offroad_count = failed_count = 0

    obs = eval_env.reset()
    ep_reward, ep_length, ep_num = 0.0, 0, 0

    while ep_num < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        ep_reward += reward[0]
        ep_length += 1

        if done[0]:
            ep_num += 1
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)

            arrive_dest = info[0].get("arrive_dest", False)
            crash       = info[0].get("crash",       False)
            out_of_road = info[0].get("out_of_road", False)

            if arrive_dest:          success_count   += 1
            if crash:                collision_count += 1
            if out_of_road:          offroad_count   += 1
            if crash or out_of_road: failed_count    += 1

            if arrive_dest:       status = "✓ SUCCESS"
            elif crash:           status = "✗ CRASH  "
            elif out_of_road:     status = "~ OFFROAD"
            else:                 status = "  TIMEOUT"

            print(
                f"  Ep {ep_num:2d} | {status} | "
                f"Reward: {ep_reward:7.2f} | "
                f"Length: {ep_length:4d}"
            )
            ep_reward, ep_length = 0.0, 0
            obs = eval_env.reset()

    n = n_episodes
    return {
        "mean_reward":     np.mean(episode_rewards),
        "std_reward":      np.std(episode_rewards),
        "mean_length":     np.mean(episode_lengths),
        "success_rate":    success_count / n,
        "crash_rate":      collision_count / n,
        "offroad_rate":    offroad_count / n,
        "survival_rate":   (n - failed_count) / n,
        "n":               n,
        "success_count":   success_count,
        "collision_count": collision_count,
        "offroad_count":   offroad_count,
    }


def print_summary(stage_name, r):
    n = r["n"]
    print(f"\n  --- Summary: {stage_name} ---")
    print(f"  Mean reward:   {r['mean_reward']:.2f} ± {r['std_reward']:.2f}")
    print(f"  Mean length:   {r['mean_length']:.1f}")
    print(f"  Success rate:  {r['success_count']}/{n}  ({r['success_rate']*100:.0f}%)")
    print(f"  Crash rate:    {r['collision_count']}/{n}  ({r['crash_rate']*100:.0f}%)")
    print(f"  Off-road rate: {r['offroad_count']}/{n}  ({r['offroad_rate']*100:.0f}%)")
    print(f"  Survival rate: {int(r['survival_rate']*n)}/{n}  ({r['survival_rate']*100:.0f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]
    phase      = next((a for a in args if a in PROFILES), "continuous")
    use_render = "--render" in args
    all_stages = "--all-stages" in args

    stage_idx = None
    if "--stage" in args:
        idx = args.index("--stage")
        if idx + 1 < len(args):
            stage_idx = int(args[idx + 1])

    profile = PROFILES[phase]
    stages  = profile["stages"]

    if stage_idx is None and not all_stages:
        stage_idx = profile["default_stage_idx"]

    stages_to_eval = stages if all_stages else [stages[stage_idx]]

    print(f"\nPhase:   {phase}")
    print(f"Model:   {profile['model_path']}")
    print(f"VecNorm: {profile['vecnorm_path']}")

    for stage in stages_to_eval:
        print(f"\n{'='*60}")
        print(f"Evaluating on: {stage['name']}")
        print(f"{'='*60}")

        eval_vec = SubprocVecEnv([
            make_eval_env(EVAL_SEED, stage, use_render=use_render)
        ])
        eval_env = VecNormalize.load(profile["vecnorm_path"], eval_vec)
        eval_env.training    = False
        eval_env.norm_reward = False

        try:
            model = PPO.load(profile["model_path"], env=eval_env, device="cpu")
        except FileNotFoundError:
            print(f"Error: model not found at {profile['model_path']}")
            print("Run train.py first.")
            eval_env.close()
            sys.exit(1)

        results = evaluate_stage(model, eval_env, N_EVAL_EPISODES)
        print_summary(stage["name"], results)
        eval_env.close()