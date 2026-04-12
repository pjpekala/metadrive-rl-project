"""
MetaDrive RL Training with SAC Algorithm
==========================================
Features:
- Multi-modal sensor support: Sparse/Dense LiDAR, BEV, RGB Camera
- Interrupt-safe training with auto-resume
- Append-style CSV logging for experiment continuity
- TensorBoard + CSV logging
- Headless server support (RunPod, etc.) with proper rendering configuration

Installation:
    pip install metadrive-simulator stable-baselines3[extra] tensorboard imageio imageio-ffmpeg

Usage:
    # Train Sparse LiDAR (baseline)
    python train_metadrive.py --modality sparse_lidar --total-timesteps 1_000_000

    # Train Dense LiDAR
    python train_metadrive.py --modality dense_lidar --total-timesteps 1_000_000

    # Train Semantic BEV (CNN) - use run_training.sh for headless servers
    ./run_training.sh train_metadrive.py --modality bev --total-timesteps 1_000_000

    # Train RGB Camera (CNN) - use run_training.sh for headless servers
    ./run_training.sh train_metadrive.py --modality rgb --total-timesteps 1_000_000

    # Resume interrupted training
    python train_metadrive.py --modality sparse_lidar

    # Evaluate a saved model (separate script)
    python evaluate_metadrive.py --model-path ./models/sparse_lidar/sac_metadrive_final.zip --modality sparse_lidar --render
"""

# =============================================================================
# HEADLESS RENDERING CONFIGURATION (must be BEFORE other imports)
# =============================================================================
# On headless servers (RunPod, etc.), we need to configure Panda3D to work
# without a physical display and handle OpenGL context limitations gracefully.
# =============================================================================

import os

# Set environment variables for headless rendering
os.environ.setdefault("PANDA_PRC_DIR", "")
os.environ.setdefault("NVIDIA_DRIVER_CAPABILITIES", "all")
os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

from panda3d.core import loadPrcFileData

# Core rendering settings for headless servers
# CRITICAL: Use EGL (p3headlessgl) instead of GLX for GPU-accelerated headless rendering
# GLX with Xvfb falls back to software rendering (llvmpipe) which is extremely slow
loadPrcFileData("", "load-display p3headlessgl")  # Use EGL for hardware acceleration
loadPrcFileData("", "notify-level-device error")  # Suppress /dev/input errors
loadPrcFileData("", "audio-library-name null")  # Disable audio
loadPrcFileData("", "gl-version 3 2")  # Use OpenGL 3.2 core profile
loadPrcFileData("", "framebuffer-srgb false")  # Disable sRGB (simpler rendering)

# =============================================================================
# SIMPLEPBR HEADLESS PATCH
# =============================================================================
# On headless servers, simplepbr's tonemapping setup can fail because the OpenGL
# context doesn't support certain features. This patch makes it fail gracefully.
# MetaDrive bundles its own simplepbr at metadrive.third_party.simplepbr
# =============================================================================

_headless_patches_applied = False


def _apply_headless_patches() -> None:
    """Apply patches for headless rendering compatibility.

    This function patches simplepbr.Pipeline.__init__ to gracefully handle
    failures in _setup_tonemapping() which can occur on headless servers
    when the OpenGL context doesn't support certain features.

    This must be called in each subprocess before creating MetaDrive environments.
    """
    global _headless_patches_applied

    if _headless_patches_applied:
        return

    try:
        # MetaDrive uses its own bundled simplepbr, not the standard package
        from metadrive.third_party import simplepbr

        original_init = simplepbr.Pipeline.__init__

        def patched_init(self, *args, **kwargs):
            try:
                original_init(self, *args, **kwargs)
            except AttributeError as e:
                if "set_shader" in str(e):
                    # Tonemapping setup failed - this is expected on headless servers
                    # Initialize minimal state to prevent crashes
                    print("[!] simplepbr tonemapping disabled (headless mode)")
                    self.use_330 = False
                    self.window = None
                    self.camera_node = None
                    self.max_lights = 8
                    self.enable_shadows = False
                    self.enable_fog = False
                    self.exposure = 1.0
                else:
                    raise

        simplepbr.Pipeline.__init__ = patched_init
        _headless_patches_applied = True

    except ImportError:
        # metadrive.third_party.simplepbr not available, no patching needed
        pass


# =============================================================================
# END HEADLESS CONFIGURATION - Regular imports follow
# =============================================================================

import argparse
import csv
import time
import signal
import atexit
import gc
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

np.float = np.float64
np.int = np.int64
np.bool = np.bool_

from metadrive.envs import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.semantic_camera import SemanticCamera
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

MODALITY_CONFIG = {
    "sparse_lidar": {
        "description": "Sparse LiDAR (80 beams, 50m)",
        "policy": "MlpPolicy",
        "num_lasers": 80,
        "lidar_distance": 50,
        "image_observation": False,
    },
    "dense_lidar": {
        "description": "Dense LiDAR (240 beams, 100m)",
        "policy": "MlpPolicy",
        "num_lasers": 240,
        "lidar_distance": 100,
        "image_observation": False,
    },
    "bev": {
        "description": "Semantic BEV (96x96)",
        "policy": "MultiInputPolicy",
        "image_observation": True,
        "camera": "topdown",
        "image_size": 96,
        "stack_size": 1,
    },
    "rgb": {
        "description": "RGB Camera (96x96)",
        "policy": "MultiInputPolicy",
        "image_observation": True,
        "camera": "rgb",
        "image_size": 96,
        "stack_size": 1,
    },
}

TRAIN_SEED_START = 0
TRAIN_SEED_COUNT = 800
EVAL_SEED_START = 800
EVAL_SEED_COUNT = 200
EVAL_SEED_END = EVAL_SEED_START + EVAL_SEED_COUNT - 1

CSV_LOG_HEADER = [
    "timestep",
    "episode",
    "mean_reward",
    "success_rate",
    "collision_rate",
    "out_of_road_rate",
    "mean_episode_length",
    "mean_speed",
]


from gymnasium import Env, spaces


class GymToGymnasiumWrapper(Env):
    metadata = {"render_modes": []}

    def __init__(self, env, modality: str):
        self.env = env
        self.modality = modality
        self.observation_space = self._convert_space(
            env.observation_space, is_observation=True
        )
        self.action_space = self._convert_space(env.action_space, is_observation=False)
        self._max_episode_steps = 2000

    def _convert_space(self, space, is_observation: bool = True):
        if isinstance(space, spaces.Dict):
            # Handle Dict observation spaces from MetaDrive
            if is_observation and self.modality in ["rgb", "bev"]:
                image_space = space["image"]
                state_space = space["state"]

                if len(image_space.shape) == 4:
                    h, w, c, stack_size = image_space.shape
                else:
                    h, w, c = image_space.shape
                    stack_size = 1

                image_shape = (c * stack_size, h, w)
                image_low = np.zeros(image_shape, dtype=np.float32)
                image_high = np.ones(image_shape, dtype=np.float32)

                state_low = (
                    np.array(state_space.low, dtype=np.float32)
                    if state_space.low is not None
                    else None
                )
                state_high = (
                    np.array(state_space.high, dtype=np.float32)
                    if state_space.high is not None
                    else None
                )

                return spaces.Dict(
                    {
                        "image": spaces.Box(
                            low=image_low,
                            high=image_high,
                            shape=image_shape,
                            dtype=np.float32,
                        ),
                        "state": spaces.Box(
                            low=state_low,
                            high=state_high,
                            shape=state_space.shape,
                            dtype=np.float32,
                        ),
                    }
                )
            else:
                # Non-image Dict observations: convert recursively
                return spaces.Dict(
                    {
                        k: self._convert_space(v, is_observation=is_observation)
                        for k, v in space.spaces.items()
                    }
                )
        elif isinstance(space, spaces.Box):
            low = (
                np.array(space.low, dtype=np.float32) if space.low is not None else None
            )
            high = (
                np.array(space.high, dtype=np.float32)
                if space.high is not None
                else None
            )

            # Only apply image transformation to observation spaces of image-based modalities
            modality_config = MODALITY_CONFIG.get(
                self.modality, MODALITY_CONFIG["sparse_lidar"]
            )
            if is_observation and modality_config.get("image_observation", False):
                # For image-based modalities, construct the proper image shape (C, H, W)
                image_size = modality_config.get("image_size", 128)
                stack_size = modality_config.get("stack_size", 1)
                num_channels = 3 * stack_size  # RGB * stack_size
                new_shape = (num_channels, image_size, image_size)

                # Reconstruct bounds for the image tensor
                low = np.zeros(new_shape, dtype=np.float32)
                high = np.ones(new_shape, dtype=np.float32)

                return spaces.Box(low=low, high=high, shape=new_shape, dtype=np.float32)

            # For non-image observations or action spaces, use the original shape
            return spaces.Box(low=low, high=high, shape=space.shape, dtype=np.float32)
        elif isinstance(space, spaces.Discrete):
            return spaces.Discrete(space.n)
        elif isinstance(space, spaces.Tuple):
            return spaces.Tuple(
                tuple(
                    self._convert_space(s, is_observation=is_observation)
                    for s in space.spaces
                )
            )
        return space

    @staticmethod
    def _format_stacked_image(image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        if len(image.shape) == 4:
            h, w, c, stack_size = image.shape
            image = np.transpose(image, (3, 2, 0, 1))
            image = image.reshape(c * stack_size, h, w)
            return np.ascontiguousarray(image)
        if len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
            return np.ascontiguousarray(image)
        raise ValueError(f"Unexpected image shape: {image.shape}")

    def reset(self, seed=None, options=None):
        result = self.env.reset(seed=seed)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        # Handle Dict observations from image modalities
        if isinstance(obs, dict) and self.modality in ["rgb", "bev"]:
            obs = {
                "image": self._format_stacked_image(obs["image"]),
                "state": np.asarray(obs["state"], dtype=np.float32),
            }
        else:
            # Existing behavior for LiDAR modalities
            obs = obs.astype(np.float32)

        return obs, info

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        else:
            obs, reward, terminated, truncated, info = result

        # Handle Dict observations from image modalities
        if isinstance(obs, dict) and self.modality in ["rgb", "bev"]:
            obs = {
                "image": self._format_stacked_image(obs["image"]),
                "state": np.asarray(obs["state"], dtype=np.float32),
            }
        else:
            # Existing behavior for LiDAR modalities
            obs = obs.astype(np.float32)

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()


class MetaDriveEnvWrapper(MetaDriveEnv):
    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None, **kwargs
    ):
        return super().reset(**kwargs)


def get_env_config(
    modality: str, is_eval: bool = False, render: bool = False
) -> dict[str, Any]:
    config = MODALITY_CONFIG.get(modality, MODALITY_CONFIG["sparse_lidar"])

    if is_eval:
        start_seed = EVAL_SEED_START
        num_scenarios = EVAL_SEED_COUNT
    else:
        start_seed = TRAIN_SEED_START
        num_scenarios = TRAIN_SEED_COUNT

    env_config: dict[str, Any] = dict(
        use_render=render,
        manual_control=False,
        start_seed=start_seed,
        num_scenarios=num_scenarios,
        traffic_density=0.1,
        horizon=2000,
        driving_reward=1.0,
        speed_reward=0.1,
        out_of_road_penalty=5.0,
        crash_vehicle_penalty=5.0,
        crash_object_penalty=5.0,
        success_reward=10.0,
        out_of_route_done=True,
        window_size=(1200, 800) if render else (800, 600),
        camera_height=3.0,
        camera_dist=8.0,
    )

    if config["image_observation"]:
        # Enable image observation mode in MetaDrive
        env_config["image_observation"] = True

        camera_type = config["camera"]
        stack_size = config.get("stack_size", 1)

        # Register the camera sensor with MetaDrive
        # IMPORTANT: Sensor name MUST match the image_source name!
        if camera_type == "rgb":
            env_config["sensors"] = {"rgb": (RGBCamera, 96, 96)}
        elif camera_type == "topdown":
            env_config["sensors"] = {"topdown": (SemanticCamera, 96, 96)}

        # Map vehicle config to image source
        env_config["vehicle_config"] = dict(
            image_source=camera_type,
            show_lidar=False,
            lidar=dict(num_lasers=0, distance=0),
            show_lane_line_detector=False,
            lane_line_detector=dict(num_lasers=0, distance=20),
            show_side_detector=False,
            side_detector=dict(num_lasers=0, distance=50),
        )

        # Configure frame stacking - MetaDrive uses ImageObservation which respects stack_size
        env_config["stack_size"] = stack_size
    else:
        env_config["vehicle_config"] = dict(
            show_lidar=True,
            lidar=dict(
                num_lasers=config["num_lasers"],
                distance=config["lidar_distance"],
            ),
            show_lane_line_detector=False,
            show_side_detector=False,
        )

    return env_config


def create_metadrive_env(
    modality: str, is_eval: bool = False, render: bool = False
) -> GymToGymnasiumWrapper:
    # Apply headless patches in each subprocess for image modalities
    _apply_headless_patches()

    config = get_env_config(modality, is_eval=is_eval, render=render)
    env = MetaDriveEnvWrapper(config=config)
    return GymToGymnasiumWrapper(env, modality=modality)


class GarbageCollectionCallback(BaseCallback):
    """Periodically trigger garbage collection to reduce memory fragmentation.

    This callback forces garbage collection every N steps to prevent memory
    fragmentation and reduce memory usage, especially important for image-based
    observations which create many temporary array objects.
    """

    def __init__(self, gc_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.gc_freq = gc_freq
        self._step_count = 0

    def _on_step(self) -> bool:
        self._step_count += 1
        if self._step_count % self.gc_freq == 0:
            gc.collect()
            if self.verbose > 0:
                print(f"[*] Garbage collection triggered at step {self._step_count}")
        return True


class MixedPrecisionCallback(BaseCallback):
    """Enable automatic mixed precision (AMP) training for SAC.

    This callback wraps the model's training operations with torch.autocast
    to use fp16/bfloat16 for forward passes while keeping gradients in fp32.
    This reduces memory usage by ~40-50% with minimal impact on training stability.

    Attributes:
        enabled: Whether AMP is actually enabled (requires CUDA)
        scaler: Gradient scaler for loss scaling in mixed precision
    """

    def __init__(self, enabled: bool = True, verbose: int = 1):
        super().__init__(verbose)
        self._enabled_flag = enabled
        self.enabled = False  # Will be set in _on_training_start
        self.scaler = None
        self._original_train = None
        self._step_count = 0

    def _on_training_start(self) -> None:
        if not self._enabled_flag:
            if self.verbose > 0:
                print("[*] Mixed precision disabled by user.")
            return

        # Check if CUDA is available
        if not torch.cuda.is_available():
            print("[!] CUDA not available, mixed precision disabled.")
            self.enabled = False
            return

        # Enable autocast for the model's train method
        self.enabled = True
        self.scaler = torch.amp.GradScaler("cuda")

        # Store original train method and replace with AMP-wrapped version
        if hasattr(self.model, "train"):
            self._original_train = self.model.train

            def amp_train(*args, **kwargs):
                with torch.amp.autocast("cuda"):
                    return self._original_train(*args, **kwargs)

            self.model.train = amp_train

        if self.verbose > 0:
            print(f"[*] Mixed precision training enabled (GradScaler active)")
            print(f"    Device: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA capability: {torch.cuda.get_device_capability(0)}")

    def _on_step(self) -> bool:
        self._step_count += 1

        # Log memory usage periodically
        if self.enabled and self.verbose > 0 and self._step_count % 5000 == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(
                f"[*] GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
            )

        return True

    def _on_training_end(self) -> None:
        if self._original_train is not None and hasattr(self.model, "train"):
            self.model.train = self._original_train
            if self.verbose > 0:
                print("[*] Restored original training method.")


class AppendCSVCallback(BaseCallback):
    def __init__(
        self,
        csv_path: str,
        log_freq: int = 4096,
        initial_timesteps: int = 0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.csv_path = Path(csv_path)
        self.log_freq = log_freq
        self.initial_timesteps = initial_timesteps
        self._last_log = initial_timesteps
        self._is_resume = initial_timesteps > 0
        self._ep_count = 0
        self._successes = []
        self._collisions = []
        self._out_of_road = []
        self._lengths = []
        self._speeds = []
        self._rewards = []

        if not self.csv_path.exists():
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(CSV_LOG_HEADER)
        else:
            if initial_timesteps > 0:
                last_ts, last_ep = self._get_last_log_entry()
                self._ep_count = last_ep
                if last_ts >= initial_timesteps and last_ts <= initial_timesteps + 5000:
                    self._last_log = last_ts
                else:
                    self._last_log = initial_timesteps
            else:
                self._last_log = 0
                self._ep_count = 0

    def _get_last_log_entry(self) -> tuple:
        try:
            with open(self.csv_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    return int(rows[-1]["timestep"]), int(rows[-1]["episode"])
        except Exception:
            pass
        return 0, 0

    def _on_step(self) -> bool:
        ts = (
            self.model.num_timesteps
            if hasattr(self, "model") and self.model is not None
            else 0
        )

        for info in self.locals.get("infos", [{}]):
            if "episode" in info:
                self._ep_count += 1
                ep_info = info["episode"]
                self._successes.append(float(info.get("arrive_dest", False)))
                self._collisions.append(float(info.get("crash", False)))
                self._out_of_road.append(float(info.get("out_of_road", False)))
                self._lengths.append(ep_info["l"])
                self._rewards.append(float(ep_info["r"]))
                self._speeds.append(float(ep_info["r"]) / max(float(ep_info["l"]), 1))

        if not self._successes:
            return True

        if ts - self._last_log < self.log_freq:
            return True

        if ts <= self._last_log:
            return True

        if self._is_resume and len(self._successes) < 100:
            return True

        mean_reward = round(float(np.mean(self._rewards[-100:])), 2)
        mean_speed = round(float(np.mean(self._speeds[-100:])), 2)
        success_rate = round(float(np.mean(self._successes[-100:])), 3)
        collision_rate = round(float(np.mean(self._collisions[-100:])), 3)
        out_of_road_rate = round(float(np.mean(self._out_of_road[-100:])), 3)
        mean_length = round(float(np.mean(self._lengths[-100:])), 1)

        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    ts,
                    self._ep_count,
                    mean_reward,
                    success_rate,
                    collision_rate,
                    out_of_road_rate,
                    mean_length,
                    mean_speed,
                ]
            )

        self.logger.record("train/mean_reward", mean_reward)
        self.logger.record("train/success_rate", success_rate)
        self.logger.record("train/collision_rate", collision_rate)
        self.logger.record("train/out_of_road_rate", out_of_road_rate)
        self.logger.record("train/mean_speed", mean_speed)
        self.logger.record("train/mean_episode_length", mean_length)

        self._last_log = ts
        return True


class InlineEvalCallback(BaseCallback):
    def __init__(
        self,
        modality: str,
        eval_freq: int = 100_000,
        eval_episodes: int = 8,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.modality = modality
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self._last_eval_step = 0
        self._eval_env = None

    def _on_training_start(self) -> None:
        if self.eval_freq <= 0 or self.eval_episodes <= 0:
            return
        self._eval_env = create_metadrive_env(self.modality, is_eval=True, render=False)

    def _run_eval(self) -> dict[str, float]:
        if self._eval_env is None:
            return {
                "success_rate": 0.0,
                "collision_rate": 0.0,
                "out_of_road_rate": 0.0,
                "mean_reward": 0.0,
                "mean_episode_length": 0.0,
            }

        rewards: list[float] = []
        lengths: list[float] = []
        successes: list[float] = []
        collisions: list[float] = []
        out_of_road: list[float] = []

        for _ in range(self.eval_episodes):
            obs, _ = self._eval_env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            info = {}

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self._eval_env.step(action)
                done = terminated or truncated
                episode_reward += float(reward)
                episode_length += 1

            rewards.append(episode_reward)
            lengths.append(float(episode_length))
            successes.append(float(info.get("arrive_dest", False)))
            collisions.append(float(info.get("crash", False)))
            out_of_road.append(float(info.get("out_of_road", False)))

        return {
            "success_rate": float(np.mean(successes)),
            "collision_rate": float(np.mean(collisions)),
            "out_of_road_rate": float(np.mean(out_of_road)),
            "mean_reward": float(np.mean(rewards)),
            "mean_episode_length": float(np.mean(lengths)),
        }

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.eval_episodes <= 0:
            return True

        current_step = self.model.num_timesteps
        if current_step - self._last_eval_step < self.eval_freq:
            return True

        metrics = self._run_eval()
        self.logger.record("eval/success_rate", metrics["success_rate"])
        self.logger.record("eval/collision_rate", metrics["collision_rate"])
        self.logger.record("eval/out_of_road_rate", metrics["out_of_road_rate"])
        self.logger.record("eval/mean_reward", metrics["mean_reward"])
        self.logger.record("eval/mean_episode_length", metrics["mean_episode_length"])
        self._last_eval_step = current_step

        if self.verbose > 0:
            print(
                "[*] Inline eval @ "
                f"{current_step:,} steps | "
                f"Success: {metrics['success_rate']:.3f} | "
                f"Out-of-road: {metrics['out_of_road_rate']:.3f}"
            )

        return True

    def _on_training_end(self) -> None:
        if self._eval_env is not None:
            self._eval_env.close()
            self._eval_env = None


class VisualizationCallback(BaseCallback):
    def __init__(
        self,
        render_interval: int = 999999,
        plot_interval: int = 100,
        modality: str = "sparse_lidar",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.render_interval = render_interval
        self.plot_interval = plot_interval
        self.modality = modality
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.episode_count = 0

        csv_path = Path("models") / modality / "training_log.csv"
        if csv_path.exists():
            try:
                with open(csv_path, "r") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        self.episode_count = int(rows[-1]["episode"])
            except Exception:
                pass

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals["rewards"][0]
        self.current_episode_length += 1

        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)

            info = self.locals.get("infos", [{}])[0]
            success = info.get("arrive_dest", False)
            self.episode_successes.append(1.0 if success else 0.0)

            self.episode_count += 1

            if self.verbose > 0:
                success_rate = np.mean(self.episode_successes[-100:]) * 100
                avg_reward = (
                    np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                )
                print(
                    f"[Episode {self.episode_count}] "
                    f"Reward: {self.current_episode_reward:.2f} | "
                    f"Length: {self.current_episode_length} | "
                    f"Success: {'Y' if success else 'N'} | "
                    f"Avg Reward (100): {avg_reward:.2f} | "
                    f"Success Rate (100): {success_rate:.1f}%"
                )

            self.current_episode_reward = 0
            self.current_episode_length = 0

        return True


def train_sac_agent(
    modality: str = "sparse_lidar",
    total_timesteps: int = 1_000_000,
    learning_rate: float = 3e-4,
    buffer_size: int = 1_000_000,
    learning_starts: int = 1000,
    batch_size: int = 512,
    train_freq: int = 1,
    gradient_steps: int = 1,
    n_envs: int = 8,
    save_dir: str = "./models",
    save_freq: int = 200_000,
    device: str = "auto",
    mixed_precision: bool = False,
    compile_model: bool = False,
    inline_eval: bool = True,
    eval_freq: int = 100_000,
    eval_episodes: int = 8,
):
    modality_config = MODALITY_CONFIG.get(modality, MODALITY_CONFIG["sparse_lidar"])
    policy_type = modality_config["policy"]
    is_image_modality = modality_config.get("image_observation", False)

    # Adjust hyperparameters for image modalities based on available resources
    # Only auto-adjust if using default values (allows CLI override)
    if is_image_modality:
        if buffer_size == 1_000_000:
            buffer_size = 100_000
            print(
                f"[*] Buffer size adjusted to {buffer_size:,} for image modality '{modality}'"
            )
        if learning_starts == 1000:
            learning_starts = 10_000
            print(
                f"[*] Learning starts adjusted to {learning_starts:,} for image modality '{modality}'"
            )
        if batch_size == 512:
            batch_size = 512
            print(
                f"[*] Batch size adjusted to {batch_size} for image modality '{modality}'"
            )
        if train_freq == 1:
            train_freq = 4
            print(
                f"[*] Train freq adjusted to {train_freq} for image modality '{modality}'"
            )
        if gradient_steps == 1:
            gradient_steps = 4
            print(
                f"[*] Gradient steps adjusted to {gradient_steps} for image modality '{modality}'"
            )
        if n_envs in [8, 12]:
            n_envs = 12
            print(
                f"[*] Parallel environments adjusted to {n_envs} for image modality '{modality}'"
            )

    save_dir = Path(save_dir) / modality
    save_dir.mkdir(parents=True, exist_ok=True)

    log_dir = save_dir / "logs"
    checkpoint_dir = save_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    csv_log_path = save_dir / "training_log.csv"

    model_path = save_dir / "sac_metadrive_final.zip"

    print("=" * 70)
    print(f"MetaDrive SAC Training - {modality_config['description']}")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Policy: {policy_type}")
    print(f"Parallel environments: {n_envs}")
    print(
        f"Training seeds: {TRAIN_SEED_START} to {TRAIN_SEED_START + TRAIN_SEED_COUNT - 1}"
    )
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Learning starts: {learning_starts:,}")
    print(f"Buffer size: {buffer_size:,}")
    print(f"Batch size: {batch_size}")
    print(f"Train freq: {train_freq}")
    print(f"Gradient steps: {gradient_steps}")
    print("=" * 70)

    print("\nCreating training environment...")
    print(f"Creating {n_envs} parallel environments...")

    # Use make_vec_env to create parallel environments
    env = make_vec_env(
        lambda: create_metadrive_env(modality, is_eval=False),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,  # Use SubprocVecEnv for process isolation
    )

    print(f"\nEnvironment created successfully with {n_envs} parallel workers!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    model_loaded = False
    start_timesteps = 0
    replay_buffer_path = save_dir / "replay_buffer.pkl"

    if model_path.exists():
        try:
            model = SAC.load(str(model_path), env=env, tensorboard_log=str(log_dir))
            start_timesteps = model.num_timesteps
            print(f"[+] Resumed from '{model_path}' at step {start_timesteps:,}")
            if replay_buffer_path.exists():
                try:
                    model.load_replay_buffer(str(replay_buffer_path))
                    print(f"[+] Replay buffer loaded from: {replay_buffer_path}")
                except Exception as rb_e:
                    print(
                        f"[!] Replay buffer load failed ({rb_e}) - using current buffer."
                    )
            model_loaded = True
        except Exception as e:
            print(f"[!] Model load failed ({e}) - starting fresh.")

    if not model_loaded:
        print("[>] Initializing new SAC model...")

        # For image-based policies, we need to set normalize_images=False
        # since our images are already in the range [0, 1] and channel-first format
        policy_kwargs = {}
        if modality_config.get("image_observation", False):
            policy_kwargs["normalize_images"] = False

        model = SAC(
            policy_type,
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=0.005,
            gamma=0.99,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            ent_coef="auto",
            verbose=1,
            tensorboard_log=str(log_dir),
            device=device,
            policy_kwargs=policy_kwargs,
            optimize_memory_usage=False,
        )

    # Compile model with torch.compile for optimized execution
    if compile_model:
        if torch.cuda.is_available():
            try:
                print("[*] Compiling model with torch.compile...")
                # Compile the policy network (actor) - disable CUDA graphs for RL compatibility
                if hasattr(model, "policy") and model.policy is not None:
                    model.policy = torch.compile(
                        model.policy,
                        mode="default",
                        fullgraph=False,
                        dynamic=True,
                        options={"triton.cudagraphs": False},
                    )
                    print("[+] Policy network compiled successfully")
                # Compile critic networks
                if hasattr(model, "critic") and model.critic is not None:
                    model.critic = torch.compile(
                        model.critic,
                        mode="default",
                        fullgraph=False,
                        dynamic=True,
                        options={"triton.cudagraphs": False},
                    )
                    print("[+] Critic network compiled successfully")
            except Exception as e:
                print(f"[!] torch.compile failed: {e}. Continuing without compilation.")
        else:
            print("[!] CUDA not available, torch.compile disabled.")

    remaining_timesteps = max(0, total_timesteps - start_timesteps)
    print(
        f"[>] Training {remaining_timesteps:,} remaining steps (budget {total_timesteps:,})"
    )

    callbacks = []

    callbacks.append(
        CheckpointCallback(
            save_freq=save_freq // n_envs,
            save_path=str(checkpoint_dir),
            name_prefix="sac_metadrive",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )
    )

    callbacks.extend(
        [
            GarbageCollectionCallback(gc_freq=5000, verbose=0),
            VisualizationCallback(modality=modality, verbose=0),
            AppendCSVCallback(
                str(csv_log_path),
                log_freq=4096,
                initial_timesteps=start_timesteps,
                verbose=1,
            ),
        ]
    )

    if inline_eval:
        callbacks.append(
            InlineEvalCallback(
                modality=modality,
                eval_freq=eval_freq,
                eval_episodes=eval_episodes,
                verbose=1,
            )
        )

    if mixed_precision:
        callbacks.append(MixedPrecisionCallback(enabled=True, verbose=1))

    save_model_called = False

    def save_model_on_exit():
        nonlocal save_model_called
        if save_model_called:
            return
        save_model_called = True
        print("\n[+] Saving model and replay buffer (on exit)...")
        try:
            model.save(str(model_path))
            replay_buffer_path = save_dir / "replay_buffer.pkl"
            model.save_replay_buffer(str(replay_buffer_path))
            print(f"[+] Model saved to: {model_path}")
            print(f"[+] Replay buffer saved to: {replay_buffer_path}")
        except Exception as e:
            print(f"[!] Error saving model: {e}")

    atexit.register(save_model_on_exit)

    def signal_handler(signum, frame):
        print(f"\n[!] Received signal {signum}, saving model...")
        save_model_on_exit()
        exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70)
    print(f"To monitor with TensorBoard: tensorboard --logdir {log_dir}")
    print("=" * 70 + "\n")

    try:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=callbacks,
            log_interval=4,
            progress_bar=True,
            reset_num_timesteps=not model_loaded,
            tb_log_name="SAC_Training",
        )
    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted by user!")
    finally:
        save_model_on_exit()
        print(f"[+] CSV log: {csv_log_path}")

    env.close()
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SAC agent on MetaDrive with multi-modal support"
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="sparse_lidar",
        choices=list(MODALITY_CONFIG.keys()),
        help="Sensor modality to use",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=1000,
        help="Number of steps before training starts (default: 1000, auto-adjusted to 20000 for image modalities)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments (auto-adjusted to 12 for image modalities)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["auto", "cuda", "cpu"],
        help="Device for training",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for training (default: 512, auto-adjusted to 2048 for image modalities)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1_000_000,
        help="Replay buffer size (default: 1M, auto-adjusted to 100k for image modalities)",
    )
    parser.add_argument(
        "--train-freq",
        type=int,
        default=1,
        help="Number of env steps between training calls (default: 1, auto-adjusted to 4 for image modalities)",
    )
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=1,
        help="Number of gradient steps per training call (default: 1, auto-adjusted to 4 for image modalities)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=100_000,
        help="Inline evaluation frequency in timesteps (set <=0 to disable)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=8,
        help="Number of held-out episodes per inline evaluation",
    )
    parser.add_argument(
        "--no-inline-eval",
        action="store_true",
        default=False,
        help="Disable inline held-out evaluation during training",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        default=False,
        help="Enable automatic mixed precision (AMP) training for reduced memory usage",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Enable torch.compile for policy and critic networks (speed optimization)",
    )

    args = parser.parse_args()

    train_sac_agent(
        modality=args.modality,
        total_timesteps=args.total_timesteps,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        n_envs=args.n_envs,
        device=args.device,
        mixed_precision=args.mixed_precision,
        compile_model=args.compile,
        inline_eval=not args.no_inline_eval,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
    )
