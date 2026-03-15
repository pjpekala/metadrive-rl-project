"""
Main training script for curriculum learning.

Usage:
    uv run -m src.curriculum_learning.train discrete
    uv run -m src.curriculum_learning.train continuous
    uv run -m src.curriculum_learning.train all        # runs discrete then continuous sequentially
"""

import sys
import os

# Handle both direct execution and module import
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.curriculum_learning.config import CONTINUOUS_CURRICULUM, DISCRETE_CURRICULUM, CURRICULA, TrainingConfig
    from src.curriculum_learning.env import make_env, make_inline_eval_env
    from src.curriculum_learning.callbacks import CurriculumCallback, CurriculumEvalCallback, EarlyStoppingCallback, CheckpointCallback
    from src.curriculum_learning.vecnorm_utils import load_or_create_vecnorm, create_vecnorm, sync_obs_rms, save_vecnorm
else:
    from .config import CONTINUOUS_CURRICULUM, DISCRETE_CURRICULUM, CURRICULA, TrainingConfig
    from .env import make_env, make_inline_eval_env
    from .callbacks import CurriculumCallback, CurriculumEvalCallback, EarlyStoppingCallback, CheckpointCallback
    from .vecnorm_utils import load_or_create_vecnorm, create_vecnorm, sync_obs_rms, save_vecnorm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv


def transfer_trunk(source_ckpt: str, source_vecnorm: str, target_env, stage_holder, train_seed: int, ppo_kwargs: dict):
    """Load a checkpoint and transfer MLP trunk weights to a new model."""
    print(f"[transfer_trunk] Loading source checkpoint: {source_ckpt}")
    
    tmp_vec = SubprocVecEnv([make_env(train_seed, 0, stage_holder)])
    tmp_env = load_or_create_vecnorm(source_vecnorm, tmp_vec, norm_obs=True, norm_reward=True)
    source_model = PPO.load(source_ckpt, env=tmp_env, device=ppo_kwargs["device"])
    trunk_state = source_model.policy.mlp_extractor.state_dict()
    obs_rms = tmp_env.obs_rms
    tmp_env.close()
    
    model = PPO("MlpPolicy", target_env, **ppo_kwargs)
    model.policy.mlp_extractor.load_state_dict(trunk_state)
    target_env.obs_rms = obs_rms
    
    print("[transfer_trunk] Trunk weights transferred. Output head initialized fresh.")
    return model


def run_training(
    curriculum,
    run_name: str,
    config: TrainingConfig,
    total_timesteps: int,
    trunk_source_ckpt: str = None,
    trunk_source_vecnorm: str = None,
):
    """Run curriculum training for a given curriculum and run name."""
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
    
    stage_holder = [curriculum[0]]
    print(f"\n[{run_name}] Starting at: {stage_holder[0].name}")
    print(f"[{run_name}] Device: {config.device} | Envs: {config.n_envs} | Steps: {total_timesteps:,}")
    
    train_vec = SubprocVecEnv([
        make_env(config.seeds.train, i, stage_holder) for i in range(config.n_envs)
    ])
    
    eval_vec = SubprocVecEnv([make_inline_eval_env(config.seeds.inline_eval, stage_holder)])
    
    if os.path.exists(checkpoint_path) and os.path.exists(vecnorm_ckpt_path):
        print(f"[{run_name}] Resuming from {checkpoint_path}")
        train_env = load_or_create_vecnorm(vecnorm_ckpt_path, train_vec, norm_obs=True, norm_reward=True)
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
            }
        )
    elif trunk_source_ckpt and os.path.exists(trunk_source_ckpt):
        print(f"[{run_name}] Transferring trunk from {trunk_source_ckpt}")
        train_env = create_vecnorm(train_vec, norm_obs=True, norm_reward=True, training=True)
        model = transfer_trunk(
            source_ckpt=trunk_source_ckpt,
            source_vecnorm=trunk_source_vecnorm,
            target_env=train_env,
            stage_holder=stage_holder,
            train_seed=config.seeds.train,
            ppo_kwargs=ppo_kwargs,
        )
    else:
        print(f"[{run_name}] No checkpoint found — training from scratch")
        train_env = create_vecnorm(train_vec, norm_obs=True, norm_reward=True, training=True)
        model = PPO("MlpPolicy", train_env, **ppo_kwargs)
    
    eval_env = create_vecnorm(eval_vec, norm_obs=True, norm_reward=False, training=False)
    sync_obs_rms(train_env, eval_env)
    
    curriculum_cb = CurriculumCallback(
        train_env=train_env,
        eval_env=eval_env,
        eval_freq=eval_freq,
        stage_holder=stage_holder,
        curriculum=curriculum,
        vecnorm_ckpt_path=vecnorm_ckpt_path,
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


def run_discrete():
    config = TrainingConfig()
    run_training(
        curriculum=DISCRETE_CURRICULUM,
        run_name="discrete",
        config=config,
        total_timesteps=config.discrete_timesteps,
    )


def run_continuous():
    config = TrainingConfig()
    
    final_ckpt = "models/discrete/final_model.zip"
    final_vecnorm = "models/discrete/final_vecnorm.pkl"
    best_ckpt = "models/best_discrete/best_model.zip"
    best_vecnorm = "models/best_discrete/vecnorm.pkl"
    
    final_exists = os.path.exists(final_ckpt) and os.path.exists(final_vecnorm)
    best_exists = os.path.exists(best_ckpt) and os.path.exists(best_vecnorm)
    
    if final_exists:
        trunk_source_ckpt = final_ckpt
        trunk_source_vecnorm = final_vecnorm
        print("\n" + "="*60)
        print("WARNING: Using FINAL discrete model for transfer")
        print("  (model from end of full training)")
        print("="*60 + "\n")
    elif best_exists:
        trunk_source_ckpt = best_ckpt
        trunk_source_vecnorm = best_vecnorm
        print("\n" + "="*60)
        print("WARNING: Using BEST discrete model for transfer")
        print("  (best eval reward, not final training)")
        print("="*60 + "\n")
    else:
        missing = []
        if os.path.exists(final_ckpt) and not os.path.exists(final_vecnorm):
            missing.append(f"  - Missing: {final_vecnorm}")
        if os.path.exists(best_ckpt) and not os.path.exists(best_vecnorm):
            missing.append(f"  - Missing: {best_vecnorm}")
        raise FileNotFoundError(
            "No complete discrete checkpoint found!\n"
            f"  Expected final: {final_ckpt} + {final_vecnorm}\n"
            f"  Expected best:  {best_ckpt} + {best_vecnorm}\n"
            + ("\n".join(missing) if missing else "Run discrete training first.")
        )
    
    run_training(
        curriculum=CONTINUOUS_CURRICULUM,
        run_name="continuous",
        config=config,
        total_timesteps=config.continuous_timesteps,
        trunk_source_ckpt=trunk_source_ckpt,
        trunk_source_vecnorm=trunk_source_vecnorm,
    )


PHASES = {
    "discrete": run_discrete,
    "continuous": run_continuous,
    "all": lambda: (run_discrete(), run_continuous()),
}


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in PHASES:
        print(f"Usage: python train.py [{' | '.join(PHASES)}]")
        sys.exit(1)
    
    PHASES[sys.argv[1]]()
