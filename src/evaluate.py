import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv

MODEL_LOAD_PATH = "models/ppo_metadrive.zip"
VECNORM_LOAD_PATH = "models/ppo_metadrive_vecnorm.pkl"

# FIX 1: Changed from 1000 to 2000 — seeds 1000-1019 now overlap with
# train.py's EVAL_SEED=5000 range is fine, but more importantly they
# also overlapped with the EvalCallback's held-out set in the old
# train.py (EVAL_SEED=1000). Use 2000 to keep this script's test
# scenarios cleanly separate from both training and EvalCallback seeds.
EVAL_SEED = 2000
NUM_EVAL_SCENARIOS = 20
N_EVAL_EPISODES = 20


def make_env(seed: int):
    def _init():
        env = SafeMetaDriveEnv(config={
            "num_scenarios": NUM_EVAL_SCENARIOS,
            "start_seed": seed,
            "use_render": False,  # Set True to watch the agent drive
            "horizon": 1000,
        })
        return env
    return _init


if __name__ == "__main__":
    # --- 1. Build eval env FIRST, then load model into it ---
    # FIX 2: You were loading the model before the env, then passing env=env
    # to PPO.load() — but the VecNormalize wrapper wasn't applied yet at that
    # point. Build and wrap the env first, then load the model into the
    # already-wrapped env so observation spaces are consistent.
    eval_env = SubprocVecEnv([make_env(EVAL_SEED)])
    eval_env = VecNormalize.load(VECNORM_LOAD_PATH, eval_env)
    eval_env.training = False    # Freeze running mean/var
    eval_env.norm_reward = False # Don't normalize reward at eval time

    try:
        model = PPO.load(MODEL_LOAD_PATH, env=eval_env)  # Pass wrapped env here
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