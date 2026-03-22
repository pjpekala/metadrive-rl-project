"""
Dense LiDAR training script for sensor experiments.

Usage:
    uv run -m src.curriculum_learning.train_dense_lidar
"""

import sys
import os

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.curriculum_learning.config import CONTINUOUS_CURRICULUM, TrainingConfig
    from src.curriculum_learning.env import make_env, make_inline_eval_env
    from src.curriculum_learning.callbacks import (
        CurriculumCallback,
        CurriculumEvalCallback,
        EarlyStoppingCallback,
    )
    from stable_baselines3.common.callbacks import CheckpointCallback
    from src.curriculum_learning.vecnorm_utils import (
        load_or_create_vecnorm,
        create_vecnorm,
        sync_obs_rms,
        save_vecnorm,
    )
else:
    from .config import CONTINUOUS_CURRICULUM, TrainingConfig
    from .env import make_env, make_inline_eval_env
    from .callbacks import (
        CurriculumCallback,
        CurriculumEvalCallback,
        EarlyStoppingCallback,
    )
    from stable_baselines3.common.callbacks import CheckpointCallback
    from .vecnorm_utils import (
        load_or_create_vecnorm,
        create_vecnorm,
        sync_obs_rms,
        save_vecnorm,
    )

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv


def make_dense_lidar_env(train_seed: int, scenario_index: int, stage_holder):
    """Create environment with dense LiDAR enabled."""
    stage_holder[0] = stage_holder[0]._replace(use_dense_lidar=True)
    return make_env(train_seed, scenario_index, stage_holder)()


def run_dense_lidar_training(
    curriculum,
    run_name: str,
    config: TrainingConfig,
    total_timesteps: int,
):
    """Run dense LiDAR curriculum training."""
    model_dir = f"models/{run_name}"
    log_path = f"logs/run_{run_name}"
    best_dir = f"models/best_{run_name}"
    ckpt_dir = f"models/checkpoints_{run_name}"
    checkpoint_path = f"{best_dir}/best_model.zip"
    vecnorm_ckpt_path = f"{best_dir}/vecnorm.pkl"
    final_model_path = f"{model_dir}/final_model"
    final_vecnorm_path = f"{model_dir}/final_vecnorm.pkl"

    for d in [model_dir, log_path, best_dir, ckpt_dir, "logs"]:
        os.makedirs(d, exist_ok=True)

    eval_freq = config.eval_freq_steps // config.n_envs

    ppo_kwargs = dict(
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        target_kl=config.target_kl,
        device=config.device,
        verbose=1,
        tensorboard_log=log_path,
        policy_kwargs=dict(net_arch=config.net_arch),
    )

    stage_holder = [curriculum[0]._replace(use_dense_lidar=True, lidar_num_lasers=360)]
    print(f"\n[{run_name}] Starting at: {stage_holder[0].name}")
    print(
        f"[{run_name}] Device: {config.device} | Envs: {config.n_envs} | Steps: {total_timesteps:,}"
    )
    print(f"[{run_name}] Observation: Dense LiDAR (360 rays)")

    train_vec = SubprocVecEnv(
        [make_env(config.seeds.train, i, stage_holder) for i in range(config.n_envs)]
    )

    eval_stage_holder = [
        curriculum[0]._replace(use_dense_lidar=True, lidar_num_lasers=360)
    ]
    eval_vec = SubprocVecEnv(
        [make_inline_eval_env(config.seeds.inline_eval, eval_stage_holder)]
    )

    if os.path.exists(checkpoint_path) and os.path.exists(vecnorm_ckpt_path):
        print(f"[{run_name}] Resuming from {checkpoint_path}")
        train_env = load_or_create_vecnorm(
            vecnorm_ckpt_path, train_vec, norm_obs=True, norm_reward=True
        )
        train_env.training = True
        train_env.norm_reward = True
        model = PPO.load(
            checkpoint_path,
            env=train_env,
            device=config.device,
            custom_objects={
                "learning_rate": ppo_kwargs["learning_rate"],
                "clip_range": ppo_kwargs["clip_range"],
                "n_steps": config.n_steps,
                "batch_size": config.batch_size,
            },
        )
    else:
        print(f"[{run_name}] Training from scratch")
        train_env = create_vecnorm(
            train_vec, norm_obs=True, norm_reward=True, training=True
        )
        model = PPO("MlpPolicy", train_env, **ppo_kwargs)

    eval_env = create_vecnorm(
        eval_vec, norm_obs=True, norm_reward=False, training=False
    )
    sync_obs_rms(train_env, eval_env)

    curriculum_cb = CurriculumCallback(
        train_env=train_env,
        eval_env=eval_env,
        eval_freq=eval_freq,
        stage_holder=stage_holder,
        curriculum=curriculum,
        vecnorm_ckpt_path=vecnorm_ckpt_path,
        curriculum_passes_required=config.curriculum_passes_required,
        min_evals_per_stage=config.min_evals_per_stage,
        verbose=1,
    )

    early_stopping_cb = EarlyStoppingCallback(
        curriculum_cb=curriculum_cb,
        patience=2,
        verbose=1,
    )

    eval_cb = CurriculumEvalCallback(
        curriculum_cb=curriculum_cb,
        early_stopping_cb=early_stopping_cb,
        eval_env=eval_env,
        best_model_save_path=best_dir,
        log_path=log_path,
        eval_freq=eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=eval_freq,
        save_path=ckpt_dir,
        name_prefix=f"ppo_{run_name}",
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[curriculum_cb, eval_cb, checkpoint_cb, early_stopping_cb],
        progress_bar=True,
        reset_num_timesteps=False,
    )
    print(f"\n[{run_name}] Training finished.")

    model.save(final_model_path)
    save_vecnorm(train_env, final_vecnorm_path)
    print(f"[{run_name}] Model:   {final_model_path}.zip")
    print(f"[{run_name}] VecNorm: {final_vecnorm_path}")

    train_env.close()
    eval_env.close()


def run_dense_lidar():
    config = TrainingConfig()
    config.device = "cpu"
    run_dense_lidar_training(
        curriculum=CONTINUOUS_CURRICULUM,
        run_name="dense_lidar",
        config=config,
        total_timesteps=config.continuous_timesteps,
    )


if __name__ == "__main__":
    run_dense_lidar()
