import os
from stable_baselines3.common.vec_env import VecNormalize


def load_or_create_vecnorm(
    vecnorm_path: str,
    venv,
    norm_obs: bool = True,
    norm_reward: bool = True,
    training: bool = True,
):
    """
    Load VecNormalize from file, or create a new one if:
    - File doesn't exist
    - Observation space shape doesn't match

    Returns: VecNormalize instance
    """
    if not os.path.exists(vecnorm_path):
        return VecNormalize(
            venv,
            norm_obs=norm_obs,
            norm_reward=norm_reward,
            clip_obs=10.0,
            clip_reward=10.0,
            training=training,
        )

    try:
        vecnorm = VecNormalize.load(vecnorm_path, venv)
        vecnorm.training = training
        return vecnorm
    except AssertionError as e:
        if "spaces must have the same shape" in str(e):
            print("  [VecNormalize] Shape mismatch - creating fresh normalizer")
            print("    (This typically happens when environment config changed)")
            return VecNormalize(
                venv,
                norm_obs=norm_obs,
                norm_reward=norm_reward,
                clip_obs=10.0,
                clip_reward=10.0,
                training=training,
            )
        raise


def load_or_create_vecnorm_for_eval(vecnorm_path: str, venv):
    """
    Load VecNormalize for evaluation, or create a new one if shape mismatch.
    """
    return load_or_create_vecnorm(
        vecnorm_path, venv, norm_obs=True, norm_reward=False, training=False
    )


def create_vecnorm(
    venv, norm_obs: bool = True, norm_reward: bool = False, training: bool = True
):
    """
    Create a VecNormalize with consistent defaults.

    Args:
        venv: The vectorized environment to wrap
        norm_obs: Whether to normalize observations
        norm_reward: Whether to normalize rewards
        training: Whether to run in training mode

    Returns: VecNormalize instance
    """
    return VecNormalize(
        venv,
        norm_obs=norm_obs,
        norm_reward=norm_reward,
        clip_obs=10.0,
        clip_reward=10.0,
        training=training,
    )


def sync_obs_rms(source_env, target_env):
    """
    Copy observation statistics from source to target environment.

    This is used to sync the running statistics from training to evaluation envs.
    """
    target_env.obs_rms = source_env.obs_rms


def save_vecnorm(vecnorm, path: str):
    """
    Save VecNormalize to file.
    """
    vecnorm.save(path)
