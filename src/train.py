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

# --- GPU setup ---
# SB3 runs the neural network on GPU but steps envs on CPU (correct —
# MetaDrive can't run on GPU). The 3060 Ti accelerates forward/backward
# passes during the gradient update phase.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# --- Hyperparameters ---
N_ENVS = 4
N_STEPS = 2048          # FIX 1: was 4096. Effective batch = N_ENVS * N_STEPS = 8192.
                        # 4096*4 = 16k steps before any update — too long, value
                        # function gets stale. 2048*4 = 8192 is the SB3 sweet spot.
BATCH_SIZE = 512        # FIX 2: was 256. Larger batch utilizes 3060 Ti better
                        # and gives more stable gradient estimates.
TOTAL_TIMESTEPS = 5_000_000

# FIX 3: was 100 scenarios. Too many too early — agent never sees the same
# map twice so it can't build intuition. Start at 20 for early wins,
# then expand to 100+ once you see >20% success around 1M steps.
NUM_TRAIN_SCENARIOS = 20
TRAIN_SEED = 0
EVAL_SEED = 5000        # FIX 4: was 1000. Seeds 1000-1100 overlap with your
                        # evaluate.py test set (start_seed=1000, 20 scenarios).
                        # Use 5000 to keep EvalCallback truly held-out.


def make_env(seed: int, scenario_index: int):
    def _init():
        env = SafeMetaDriveEnv(config={
            "num_scenarios": NUM_TRAIN_SCENARIOS,
            "start_seed": seed + scenario_index * NUM_TRAIN_SCENARIOS,
            "use_render": False,
            "horizon": 1000,
        })
        return env
    return _init


class SyncVecNormCallback(BaseCallback):
    """
    FIX 5 — THE PRIMARY CAUSE OF 0% SUCCESS RATE.

    SB3's EvalCallback does NOT automatically sync VecNormalize stats between
    train_env and eval_env. Your eval_env was constructed with training=False,
    meaning it had zero/uninitialized running stats. The model received
    completely unnormalized observations at eval time — totally different
    scale from training — so it looked like 0% success even if the policy
    had actually learned something.

    This callback syncs obs_rms from train_env -> eval_env before each
    evaluation, then freezes eval_env so stats don't drift during eval.
    """
    def __init__(self, train_env, eval_env, eval_freq, verbose=0):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self.eval_env.obs_rms = self.train_env.obs_rms
            self.eval_env.training = False
        return True


if __name__ == "__main__":
    # --- 1. Training env ---
    train_env = SubprocVecEnv([make_env(TRAIN_SEED, i) for i in range(N_ENVS)])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    # --- 2. Eval env ---
    # training=True here — stats will be overwritten by SyncVecNormCallback
    # before each eval anyway, so initial value doesn't matter.
    eval_env = SubprocVecEnv([make_env(EVAL_SEED, 0)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=True,
    )

    EVAL_FREQ = 25_000 // N_ENVS

    sync_cb = SyncVecNormCallback(train_env, eval_env, eval_freq=EVAL_FREQ)

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000 // N_ENVS,   # FIX 6: was 5000 — checkpointing every
                                        # 5k steps creates hundreds of files and
                                        # noticeably slows training. 100k is enough.
        save_path="models/checkpoints/",
        name_prefix=MODEL_NAME,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models/best/",
        log_path=LOG_PATH,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=10,            # FIX 7: was 5 — too noisy for a meaningful
                                        # success rate signal, 10 minimum.
        deterministic=True,
    )

    # --- 3. PPO model ---
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=DEVICE,      # <-- GPU. Env stepping stays on CPU automatically.
        verbose=1,
        tensorboard_log=LOG_PATH,
    )

    # --- 4. Train ---
    print(f"Training: {TOTAL_TIMESTEPS:,} steps | {N_ENVS} envs | "
          f"{NUM_TRAIN_SCENARIOS} scenarios | device={DEVICE}")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[sync_cb, checkpoint_cb, eval_cb],
        progress_bar=True,
    )
    print("--- Training Finished ---")

    # --- 5. Save ---
    model.save(MODEL_SAVE_PATH)
    train_env.save(VECNORM_SAVE_PATH)
    print(f"Model: {MODEL_SAVE_PATH}.zip")
    print(f"VecNorm: {VECNORM_SAVE_PATH}")

    train_env.close()
    eval_env.close()