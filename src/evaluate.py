import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv

MODEL_LOAD_PATH = "models/best/best_model.zip"
VECNORM_LOAD_PATH = "models/best/vecnorm.pkl"

# --- Seed ranges (must not overlap with train.py) ---
# train.py TRAIN_SEED: 0    → scenarios 0–79
# train.py EVAL_SEED:  5000 → inline EvalCallback
# This script EVAL_SEED: 2000 → final held-out evaluation
EVAL_SEED = 2000
NUM_EVAL_SCENARIOS = 20
N_EVAL_EPISODES = 20        # More episodes than inline eval (5) for reliable metrics


def make_env(seed: int, scenario_index: int = 0):
    """Matches train.py signature for consistency."""
    def _init():
        env = SafeMetaDriveEnv(config={
            "num_scenarios": NUM_EVAL_SCENARIOS,
            "start_seed": seed + scenario_index * NUM_EVAL_SCENARIOS,
            "use_render": False,    # Set True to watch the agent drive
            "horizon": 1000,
        })
        return env
    return _init


if __name__ == "__main__":
    # --- 1. Build env first, then load model into it ---
    # VecNormalize must be applied before PPO.load() so obs spaces are consistent.
    eval_env = SubprocVecEnv([make_env(EVAL_SEED)])
    eval_env = VecNormalize.load(VECNORM_LOAD_PATH, eval_env)
    eval_env.training = False       # Freeze running stats — don't update during eval
    eval_env.norm_reward = False    # Raw rewards for interpretable metrics

    try:
        model = PPO.load(MODEL_LOAD_PATH, env=eval_env, device="cpu")
        print(f"Model loaded from {MODEL_LOAD_PATH}")
    except FileNotFoundError:
        print(f"Error: Model not found at {MODEL_LOAD_PATH}")
        print("Please run train.py first.")
        eval_env.close()
        exit()

    # --- 2. Run evaluation episodes ---
    print(f"--- Starting Evaluation ({N_EVAL_EPISODES} episodes) ---")

    episode_rewards = []
    episode_lengths = []
    success_count = 0
    collision_count = 0

    obs = eval_env.reset()
    ep_reward = 0
    ep_length = 0
    episodes_done = 0

    while episodes_done < N_EVAL_EPISODES:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)

        ep_reward += reward[0]
        ep_length += 1

        if done[0]:
            episodes_done += 1
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)

            arrive_dest = info[0].get("arrive_dest", False)
            crash = info[0].get("crash", False)
            out_of_road = info[0].get("out_of_road", False)

            if arrive_dest:
                success_count += 1
            if crash:
                collision_count += 1

            print(
                f"Episode {episodes_done:2d} | "
                f"Reward: {ep_reward:7.2f} | "
                f"Length: {ep_length:4d} | "
                f"Success: {arrive_dest} | "
                f"Crash: {crash} | "
                f"Off-road: {out_of_road}"
            )

            ep_reward = 0
            ep_length = 0
            obs = eval_env.reset()

    # --- 3. Summary ---
    print("\n--- Evaluation Summary ---")
    print(f"Episodes:          {N_EVAL_EPISODES}")
    print(f"Mean reward:       {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean episode len:  {np.mean(episode_lengths):.1f}")
    print(f"Success rate:      {success_count / N_EVAL_EPISODES * 100:.1f}%  ({success_count}/{N_EVAL_EPISODES})")
    print(f"Collision rate:    {collision_count / N_EVAL_EPISODES * 100:.1f}%  ({collision_count}/{N_EVAL_EPISODES})")

    eval_env.close()