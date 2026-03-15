"""
Standalone evaluation script.

Loads a trained model and evaluates it on held-out scenarios.

Usage:
    uv run -m src.curriculum_learning.evaluate                          # continuous, final stage
    uv run -m src.curriculum_learning.evaluate discrete                 # discrete model
    uv run -m src.curriculum_learning.evaluate continuous --stage 2     # specific stage
    uv run -m src.curriculum_learning.evaluate continuous --all-stages # sweep all stages
    uv run -m src.curriculum_learning.evaluate continuous --render      # watch it drive
"""

import sys
import os

# Handle both direct execution and module import
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.curriculum_learning.config import CURRICULA, Stage, Seeds
    from src.curriculum_learning.env import make_held_out_eval_env
    from src.curriculum_learning.vecnorm_utils import load_or_create_vecnorm_for_eval
else:
    from .config import CURRICULA, Stage, Seeds
    from .env import make_held_out_eval_env
    from .vecnorm_utils import load_or_create_vecnorm_for_eval

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


PROFILES = {
    "discrete": {
        "model_path": "models/discrete/final_model.zip",
        "fallback_model_path": "models/best_discrete/best_model.zip",
        "vecnorm_path": "models/discrete/final_vecnorm.pkl",
        "fallback_vecnorm_path": "models/best_discrete/vecnorm.pkl",
    },
    "continuous": {
        "model_path": "models/continuous/final_model.zip",
        "fallback_model_path": "models/best_continuous/best_model.zip",
        "vecnorm_path": "models/continuous/final_vecnorm.pkl",
        "fallback_vecnorm_path": "models/best_continuous/vecnorm.pkl",
    },
}

N_EVAL_EPISODES = 20


def evaluate_stage(model, eval_env, n_episodes: int) -> dict:
    """Run evaluation on a single stage and return metrics."""
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
            crash = info[0].get("crash", False)
            out_of_road = info[0].get("out_of_road", False)
            
            if arrive_dest:
                success_count += 1
            if crash:
                collision_count += 1
            if out_of_road:
                offroad_count += 1
            if crash or out_of_road:
                failed_count += 1
            
            if arrive_dest:
                status = "✓ SUCCESS"
            elif crash:
                status = "✗ CRASH  "
            elif out_of_road:
                status = "~ OFFROAD"
            else:
                status = "  TIMEOUT"
            
            print(
                f"  Ep {ep_num:2d} | {status} | "
                f"Reward: {ep_reward:7.2f} | "
                f"Length: {ep_length:4d}"
            )
            ep_reward, ep_length = 0.0, 0
            obs = eval_env.reset()
    
    n = n_episodes
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "success_rate": success_count / n,
        "crash_rate": collision_count / n,
        "offroad_rate": offroad_count / n,
        "survival_rate": (n - failed_count) / n,
        "n": n,
        "success_count": success_count,
        "collision_count": collision_count,
        "offroad_count": offroad_count,
    }


def print_summary(stage_name: str, r: dict):
    """Print evaluation summary."""
    n = r["n"]
    print(f"\n  --- Summary: {stage_name} ---")
    print(f"  Mean reward:   {r['mean_reward']:.2f} ± {r['std_reward']:.2f}")
    print(f"  Mean length:   {r['mean_length']:.1f}")
    print(f"  Success rate:  {r['success_count']}/{n}  ({r['success_rate']*100:.0f}%)")
    print(f"  Crash rate:    {r['collision_count']}/{n}  ({r['crash_rate']*100:.0f}%)")
    print(f"  Off-road rate: {r['offroad_count']}/{n}  ({r['offroad_rate']*100:.0f}%)")
    print(f"  Survival rate: {int(r['survival_rate']*n)}/{n}  ({r['survival_rate']*100:.0f}%)")


def main():
    """Entry point for evaluation."""
    args = sys.argv[1:]
    phase = next((a for a in args if a in PROFILES), "continuous")
    use_render = "--render" in args
    all_stages = "--all-stages" in args
    
    stage_idx = None
    if "--stage" in args:
        idx = args.index("--stage")
        if idx + 1 >= len(args):
            print("Error: --stage requires a value (e.g., --stage 1)")
            sys.exit(1)
        try:
            stage_idx = int(args[idx + 1]) - 1
        except ValueError:
            print(f"Error: --stage value must be an integer, got '{args[idx + 1]}'")
            sys.exit(1)
    
    profile = PROFILES[phase]
    curriculum = CURRICULA[phase]
    
    if stage_idx is not None and (stage_idx < 0 or stage_idx >= len(curriculum)):
        print(f"Error: stage must be between 1 and {len(curriculum)}")
        sys.exit(1)
    
    if stage_idx is None and not all_stages:
        stage_idx = len(curriculum) - 1
    
    stages_to_eval = curriculum if all_stages else [curriculum[stage_idx]]
    
    if all_stages:
        print("\n" + "="*60)
        print("WARNING: Evaluating FINAL model on all stages")
        print("  This tests generalization, not curriculum progression.")
        print("  Each stage was trained with different difficulty.")
        print("="*60 + "\n")
    
    # Determine which model/vecnorm to use (prefer final, fallback to best)
    model_path = profile["model_path"]
    vecnorm_path = profile["vecnorm_path"]
    
    if os.path.exists(model_path) and os.path.exists(vecnorm_path):
        pass  # Use primary paths
    elif os.path.exists(profile["fallback_model_path"]) and os.path.exists(profile["fallback_vecnorm_path"]):
        model_path = profile["fallback_model_path"]
        vecnorm_path = profile["fallback_vecnorm_path"]
        print("\n" + "="*60)
        print("WARNING: Using BEST model for evaluation")
        print("  (final model not found)")
        print("="*60 + "\n")
    else:
        print(f"Error: No model found for {phase}")
        print(f"  Tried: {profile['model_path']}")
        print(f"  Tried: {profile['fallback_model_path']}")
        print("Run train.py first.")
        sys.exit(1)
    
    print(f"\nPhase:   {phase}")
    print(f"Model:   {model_path}")
    print(f"VecNorm: {vecnorm_path}")
    
    for stage in stages_to_eval:
        print(f"\n{'='*60}")
        print(f"Evaluating on: {stage.name}")
        print(f"  Training scenarios: {stage.num_scenarios_train}")
        print(f"  Evaluation scenarios: {stage.num_scenarios_eval}")
        print(f"{'='*60}")
        
        eval_vec = DummyVecEnv([
            make_held_out_eval_env(seed=Seeds.held_out_eval, stage=stage, use_render=use_render)
        ])
        eval_env = load_or_create_vecnorm_for_eval(vecnorm_path, eval_vec)
        eval_env.training = False
        eval_env.norm_reward = False
        
        try:
            model = PPO.load(model_path, env=eval_env, device="cpu")
        except FileNotFoundError:
            print(f"Error: model not found at {model_path}")
            print("Run train.py first.")
            eval_env.close()
            sys.exit(1)
        
        results = evaluate_stage(model, eval_env, N_EVAL_EPISODES)
        print_summary(stage.name, results)
        eval_env.close()


if __name__ == "__main__":
    main()
