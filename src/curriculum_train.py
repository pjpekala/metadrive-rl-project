import warnings
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("models/checkpoints", exist_ok=True)
os.makedirs("models/best", exist_ok=True)

MODEL_NAME = "ppo_metadrive"
LOG_PATH = "logs/run_curriculum"
MODEL_SAVE_PATH = f"models/{MODEL_NAME}"
VECNORM_SAVE_PATH = f"models/{MODEL_NAME}_vecnorm.pkl"

DEVICE = "cpu"
print(f"Using device: {DEVICE}")

N_ENVS = 4
N_STEPS = 2048
BATCH_SIZE = 512
TOTAL_TIMESTEPS = 5_000_000
TRAIN_SEED = 0
EVAL_SEED = 5000

# ---------------------------------------------------------------------------
# Curriculum stages — agent advances when smoothed eval reward exceeds
# the threshold for that stage. Each stage increases map complexity,
# number of scenarios, and traffic density progressively.
#
# Stage 0: Straight roads only, no traffic, few maps — agent learns basic
#          throttle/steering and lane-keeping with no distractions
# Stage 1: Add curves, still no traffic — agent learns to handle turns
# Stage 2: Mixed road types, light traffic — add dynamic obstacles
# Stage 3: Full random maps, moderate traffic — close to final eval conditions
# ---------------------------------------------------------------------------
CURRICULUM = [
    {
        "name":             "Stage 0 — Straight roads, no traffic",
        "map":              3,          # 3 straight blocks
        "num_scenarios":    5,          # very few maps — agent gets repetitions
        "traffic_density":  0.0,        # no other vehicles
        "reward_threshold": 300,        # advance when smoothed reward > 300
    },
    {
        "name":             "Stage 1 — Curves added, no traffic",
        "map":              4,          # 4 blocks including curves
        "num_scenarios":    10,
        "traffic_density":  0.0,
        "reward_threshold": 250,
    },
    {
        "name":             "Stage 2 — Mixed roads, light traffic",
        "map":              5,
        "num_scenarios":    15,
        "traffic_density":  0.1,        # sparse traffic
        "reward_threshold": 200,
    },
    {
        "name":             "Stage 3 — Random maps, moderate traffic",
        "map":              7,
        "num_scenarios":    20,
        "traffic_density":  0.2,
        "reward_threshold": float("inf"),  # final stage — no advancement
    },
]


class RobustMetaDriveEnv(SafeMetaDriveEnv):
    """
    Thin wrapper that catches MetaDrive internal errors during step() and
    treats them as episode terminations rather than crashing the subprocess.
    Prevents BrokenPipeError from propagating to the main training process
    when a specific map or scenario triggers a MetaDrive bug.
    """
    def step(self, action):
        try:
            return super().step(action)
        except Exception as e:
            print(f"[RobustEnv] MetaDrive step() error: {e} — treating as termination")
            obs, _, _, _, info = super().reset(), 0.0, True, False, {}
            return obs, -10.0, True, False, info


def make_env(seed: int, scenario_index: int, stage_config: dict):
    """Factory — rebuilds env with current curriculum stage config."""
    def _init():
        env = RobustMetaDriveEnv(config={
            "num_scenarios":        stage_config["num_scenarios"],
            "start_seed":           seed + scenario_index * stage_config["num_scenarios"],
            "map":                  stage_config["map"],
            "traffic_density":      stage_config["traffic_density"],
            "out_of_road_penalty":  10.0,   # stronger than default 5.0
            "crash_vehicle_penalty": 10.0,
            "use_render":           False,
            "horizon":              1000,
        })
        return env
    return _init


def make_eval_env(seed: int, stage_config: dict):
    """Separate factory for eval env — uses fixed seed range."""
    def _init():
        env = RobustMetaDriveEnv(config={
            "num_scenarios":        stage_config["num_scenarios"],
            "start_seed":           seed,
            "map":                  stage_config["map"],
            "traffic_density":      stage_config["traffic_density"],
            "out_of_road_penalty":  10.0,
            "crash_vehicle_penalty": 10.0,
            "use_render":           False,
            "horizon":              1000,
        })
        return env
    return _init


class CurriculumCallback(BaseCallback):
    """
    Monitors smoothed eval reward and advances the curriculum when the
    agent exceeds the current stage's reward threshold.

    When advancing:
    - Tears down and rebuilds train_env and eval_env with the new stage config
    - Re-wraps with VecNormalize, copying running stats from the old wrapper
      so normalization continues smoothly without resetting
    - Updates the model's env reference
    - Logs the stage transition to TensorBoard
    """
    def __init__(self, train_env_ref, eval_env_ref, eval_freq,
                 model_ref, curriculum, train_seed, eval_seed,
                 n_envs, verbose=1):
        super().__init__(verbose)
        self.train_env = train_env_ref
        self.eval_env = eval_env_ref
        self.eval_freq = eval_freq
        self.model = model_ref
        self.curriculum = curriculum
        self.train_seed = train_seed
        self.eval_seed = eval_seed
        self.n_envs = n_envs
        self.current_stage = 0
        self.reward_history = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0 or self.n_calls == 1:
            # Sync and save vecnorm
            self.eval_env[0].obs_rms = self.train_env[0].obs_rms
            self.eval_env[0].training = False
            self.train_env[0].save("models/best/vecnorm.pkl")

        return True

    def on_eval_end(self, mean_reward):
        """Called externally by CurriculumEvalCallback after each eval."""
        self.reward_history.append(mean_reward)

        stage = self.curriculum[self.current_stage]
        if self.verbose:
            print(f"\n[Curriculum] Stage {self.current_stage}: "
                  f"{stage['name']} | "
                  f"Reward: {mean_reward:.1f} | "
                  f"Threshold: {stage['reward_threshold']}")

        if (mean_reward >= stage["reward_threshold"] and
                self.current_stage < len(self.curriculum) - 1):
            self.current_stage += 1
            self._advance_stage()

    def _advance_stage(self):
        import time

        stage = self.curriculum[self.current_stage]
        print(f"\n{'='*60}")
        print(f"[Curriculum] ADVANCING to Stage {self.current_stage}: {stage['name']}")
        print(f"{'='*60}\n")

        # Save old normalization stats before rebuilding envs
        old_obs_rms = self.train_env[0].obs_rms
        old_ret_rms = self.train_env[0].ret_rms

        # Close old envs — use try/except so a partially-crashed env
        # doesn't block the rebuild. Give subprocesses time to exit cleanly
        # before spinning up new ones (avoids BrokenPipeError on Linux).
        for env_ref in [self.train_env, self.eval_env]:
            try:
                env_ref[0].close()
            except Exception as e:
                print(f"[Curriculum] Warning during env close: {e}")
        time.sleep(2)  # Let subprocess workers fully exit

        # Rebuild train env with new stage config
        new_train_vec = SubprocVecEnv([
            make_env(self.train_seed, i, stage) for i in range(self.n_envs)
        ])
        new_train_env = VecNormalize(
            new_train_vec,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )
        # Carry over running stats so normalization doesn't reset
        new_train_env.obs_rms = old_obs_rms
        new_train_env.ret_rms = old_ret_rms

        # Rebuild eval env
        new_eval_vec = SubprocVecEnv([make_eval_env(self.eval_seed, stage)])
        new_eval_env = VecNormalize(
            new_eval_vec,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            training=False,
        )
        new_eval_env.obs_rms = old_obs_rms

        # Update references
        self.train_env[0] = new_train_env
        self.eval_env[0] = new_eval_env
        self.model.set_env(new_train_env)


class CurriculumEvalCallback(EvalCallback):
    """
    Wraps EvalCallback and notifies CurriculumCallback ONLY when a real
    evaluation has just completed (i.e. last_mean_reward changed this step).
    Without this guard, on_eval_end fires every env step — thousands of
    times per second — producing the -inf spam and burning CPU.
    """
    def __init__(self, curriculum_cb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum_cb = curriculum_cb
        self._prev_mean_reward = None   # Track last reported reward

    def _on_step(self) -> bool:
        result = super()._on_step()
        # last_mean_reward only changes when EvalCallback actually ran an eval.
        # By comparing to _prev_mean_reward we fire on_eval_end exactly once
        # per completed evaluation, not once per env step.
        if (self.last_mean_reward is not None and
                self.last_mean_reward != self._prev_mean_reward):
            self._prev_mean_reward = self.last_mean_reward
            self.curriculum_cb.on_eval_end(self.last_mean_reward)
        return result


if __name__ == "__main__":
    stage = CURRICULUM[0]
    print(f"Starting curriculum at: {stage['name']}")

    # Mutable containers so CurriculumCallback can swap envs in-place
    train_env_ref = [None]
    eval_env_ref = [None]

    train_vec = SubprocVecEnv([make_env(TRAIN_SEED, i, stage) for i in range(N_ENVS)])
    train_env = VecNormalize(
        train_vec,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    train_env_ref[0] = train_env

    eval_vec = SubprocVecEnv([make_eval_env(EVAL_SEED, stage)])
    eval_env = VecNormalize(
        eval_vec,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=True,
    )
    eval_env_ref[0] = eval_env

    # Load from existing checkpoint if available, else train from scratch
    CHECKPOINT_PATH = "models/best/best_model.zip"
    VECNORM_PATH = "models/best/vecnorm.pkl"

    if os.path.exists(CHECKPOINT_PATH) and os.path.exists(VECNORM_PATH):
        print(f"Warm-starting from {CHECKPOINT_PATH}")
        train_env_ref[0] = VecNormalize.load(VECNORM_PATH, train_vec)
        train_env_ref[0].training = True
        train_env_ref[0].norm_reward = True
        train_env = train_env_ref[0]
        model = PPO.load(
            CHECKPOINT_PATH,
            env=train_env,
            device=DEVICE,
            custom_objects={
                "learning_rate": 1e-4,
                "clip_range": 0.1,
                "n_steps": N_STEPS,
                "batch_size": BATCH_SIZE,
            }
        )
    else:
        print("No checkpoint found — training from scratch")
        model = PPO(
            "MlpPolicy",
            train_env,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=10,
            learning_rate=1e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device=DEVICE,
            verbose=1,
            tensorboard_log=LOG_PATH,
        )

    EVAL_FREQ = 100_000 // N_ENVS

    curriculum_cb = CurriculumCallback(
        train_env_ref=train_env_ref,
        eval_env_ref=eval_env_ref,
        eval_freq=EVAL_FREQ,
        model_ref=model,
        curriculum=CURRICULUM,
        train_seed=TRAIN_SEED,
        eval_seed=EVAL_SEED,
        n_envs=N_ENVS,
        verbose=1,
    )

    eval_cb = CurriculumEvalCallback(
        curriculum_cb=curriculum_cb,
        eval_env=eval_env_ref[0],
        best_model_save_path="models/best/",
        log_path=LOG_PATH,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=5,
        deterministic=True,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000 // N_ENVS,
        save_path="models/checkpoints/",
        name_prefix=MODEL_NAME,
    )

    print(f"Training: {TOTAL_TIMESTEPS:,} steps | {N_ENVS} envs | device={DEVICE}")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[curriculum_cb, eval_cb, checkpoint_cb],
        progress_bar=True,
        reset_num_timesteps=False,  # Continue timestep count from checkpoint
    )
    print("--- Training Finished ---")

    model.save(MODEL_SAVE_PATH)
    train_env_ref[0].save(VECNORM_SAVE_PATH)
    print(f"Model:   {MODEL_SAVE_PATH}.zip")
    print(f"VecNorm: {VECNORM_SAVE_PATH}")

    train_env_ref[0].close()
    eval_env_ref[0].close()