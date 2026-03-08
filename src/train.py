import warnings
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("models/checkpoints", exist_ok=True)
os.makedirs("models/best", exist_ok=True)

MODEL_NAME = "ppo_metadrive"
LOG_PATH = "logs"
MODEL_SAVE_PATH = f"models/{MODEL_NAME}"
VECNORM_SAVE_PATH = f"models/{MODEL_NAME}_vecnorm.pkl"

DEVICE = "cpu"                  # MLP policy — CPU is faster than GPU for this
print(f"Using device: {DEVICE}")

# --- Seed ranges (must not overlap) ---
# TRAIN_SEED:  0     → covers scenarios 0–79   (20 scenarios × 4 envs)
# EVAL_SEED:   5000  → inline EvalCallback held-out set
# (evaluate.py uses 2000 — separate from both)
N_ENVS = 4
N_STEPS = 2048
BATCH_SIZE = 512
TOTAL_TIMESTEPS = 5_000_000
NUM_TRAIN_SCENARIOS = 10
TRAIN_SEED = 0
EVAL_SEED = 5000


def make_env(seed: int, scenario_index: int):
    """Factory for SubprocVecEnv. Each worker gets a non-overlapping seed range."""
    def _init():
        env = SafeMetaDriveEnv(config={
            "num_scenarios": NUM_TRAIN_SCENARIOS,
            "start_seed": seed + scenario_index * NUM_TRAIN_SCENARIOS,
            "use_render": False,
            "horizon": 1000,
            "out_of_road_penalty": 10.0, # up from 5.0 default
            "crash_vehicle_penalty": 10.0 # up from 5.0 default
        })
        return env
    return _init


class SyncAndSaveVecNormCallback(BaseCallback):
    """
    Fires on the same frequency as EvalCallback:
    1. Syncs obs_rms from train_env -> eval_env so inline eval sees correctly
       normalized observations (fixes the 0% success rate issue).
    2. Saves a vecnorm snapshot to models/best/vecnorm.pkl so evaluate.py
       can load it at any point mid-run without waiting for training to finish.
    """
    def __init__(self, train_env, eval_env, eval_freq, verbose=0):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0 or self.n_calls == 1:
            self.eval_env.obs_rms = self.train_env.obs_rms
            self.eval_env.training = False
            self.train_env.save("models/best/vecnorm.pkl")
        return True


if __name__ == "__main__":
    # --- Training env ---
    train_env = SubprocVecEnv([make_env(TRAIN_SEED, i) for i in range(N_ENVS)])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    # --- Inline eval env (stats synced by callback before each eval) ---
    eval_env = SubprocVecEnv([make_env(EVAL_SEED, 0)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # Never normalize reward at eval time
        clip_obs=10.0,
        training=True,      # Will be frozen by SyncAndSaveVecNormCallback
    )

    EVAL_FREQ = 100_000 // N_ENVS   # Every 100k env steps — reduces training pauses

    sync_save_cb = SyncAndSaveVecNormCallback(train_env, eval_env, eval_freq=EVAL_FREQ)

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000 // N_ENVS,
        save_path="models/checkpoints/",
        name_prefix=MODEL_NAME,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models/best/",
        log_path=LOG_PATH,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=5,      # Kept low to minimize training pauses
                                # Final reported numbers use evaluate.py (20 episodes)
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=10,
        learning_rate=1e-4,   # was 3e-4
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,       # was 0.2 
        ent_coef=0.01,          # 0.05 caused std to grow unboundedly — back to 0.01
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=DEVICE,
        verbose=1,
        tensorboard_log=LOG_PATH,
    )

    print(f"Training: {TOTAL_TIMESTEPS:,} steps | {N_ENVS} envs | "
          f"{NUM_TRAIN_SCENARIOS} scenarios | device={DEVICE}")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[sync_save_cb, eval_cb, checkpoint_cb],  # sync_save_cb must be first
        progress_bar=True,
    )
    print("--- Training Finished ---")

    model.save(MODEL_SAVE_PATH)
    train_env.save(VECNORM_SAVE_PATH)
    print(f"Model:   {MODEL_SAVE_PATH}.zip")
    print(f"VecNorm: {VECNORM_SAVE_PATH}")

    train_env.close()
    eval_env.close()