import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

MODEL_NAME = "ppo_metadrive"
LOG_PATH = "logs"
MODEL_SAVE_PATH = f"models/{MODEL_NAME}"
VECNORM_SAVE_PATH = f"models/{MODEL_NAME}_vecnorm.pkl"

N_ENVS = 4             # Parallel envs — scales effective steps/update
N_STEPS = 4096         # Steps per env per rollout (effective batch = N_ENVS * N_STEPS)
TOTAL_TIMESTEPS = 5_000_000  # Increased for better convergence
# Training parameters - increased scenarios for better diversity + more training steps
NUM_TRAIN_SCENARIOS = 100   # More diverse scenarios for training
TRAIN_SEED = 0
EVAL_SEED = 1000            # Held-out seed for EvalCallback (different from training)




def make_env(seed: int, scenario_index: int):
    """Factory function — required for SubprocVecEnv."""
    def _init():
        env = SafeMetaDriveEnv(config={
            "num_scenarios": NUM_TRAIN_SCENARIOS,
            "start_seed": seed + scenario_index * NUM_TRAIN_SCENARIOS,
            "use_render": False,
            # Ensure consistent episode length for rollout buffer
            "horizon": 1000,
        })
        return env
    return _init


if __name__ == "__main__":
    # --- 1. Vectorized + normalized training env ---
    train_env = SubprocVecEnv([make_env(TRAIN_SEED, i) for i in range(N_ENVS)])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,       # Normalize observations to ~N(0,1)
        norm_reward=True,    # Normalize rewards — critical for stable value estimates
        clip_obs=10.0,       # Clip extreme obs values
        clip_reward=10.0,
    )

    # --- 2. Separate eval env (frozen normalizer stats, no reward norm) ---
    eval_env = SubprocVecEnv([make_env(EVAL_SEED, 0)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,   # Don't normalize reward at eval time
        clip_obs=10.0,
        training=False,      # Frozen — will sync stats from train_env below
    )

    # --- 3. PPO with MetaDrive-tuned hyperparameters ---
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=N_STEPS,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,       # Entropy bonus — encourages exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=LOG_PATH,
    )

    # --- 4. Callbacks ---
    checkpoint_cb = CheckpointCallback(
        save_freq=5000,  # Save every 5k steps for faster convergence tracking
        save_path="models/checkpoints/",
        name_prefix=MODEL_NAME,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models/best/",
        log_path=LOG_PATH,
        eval_freq=10000,  # Evaluate less frequently (every 10k steps)
        n_eval_episodes=5,   # Faster evaluation with fewer episodes
        deterministic=True,
    )

    # --- 5. Train ---
    print(f"--- Starting Training ({TOTAL_TIMESTEPS:,} steps, {NUM_TRAIN_SCENARIOS} scenarios, 4 envs) ---")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )
    print("--- Training Finished ---")

    # --- 6. Save model AND normalizer stats together ---
    model.save(MODEL_SAVE_PATH)
    train_env.save(VECNORM_SAVE_PATH)   # Must save this — eval.py needs it
    print(f"Model saved to {MODEL_SAVE_PATH}.zip")
    print(f"VecNormalize stats saved to {VECNORM_SAVE_PATH}")

    train_env.close()
    eval_env.close()