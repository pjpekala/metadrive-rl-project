"""
MetaDrive RL Training with SAC Algorithm
==========================================
Features:
- Multi-modal sensor support: Sparse/Dense LiDAR, BEV, RGB Camera
- Interrupt-safe training with auto-resume
- Append-style CSV logging for experiment continuity
- TensorBoard + CSV logging
- Headless server support (RunPod, etc.) with proper rendering configuration
- Hardware-aware presets (desktop / runpod) for image modalities

Installation:
    pip install metadrive-simulator stable-baselines3[extra] tensorboard imageio imageio-ffmpeg

Usage:
    # Train Sparse LiDAR (baseline)
    python train_metadrive.py --modality sparse_lidar --total-timesteps 1_000_000

    # Train Dense LiDAR
    python train_metadrive.py --modality dense_lidar --total-timesteps 1_000_000

    # Train BEV (explicit image settings, desktop-safe)
    ./run_training.sh train_metadrive.py --modality bev --n-envs 1 --buffer-size 40000 --batch-size 128 --learning-starts 3000 --train-freq 8 --gradient-steps 2 --learning-rate 1e-4 --vec-env dummy --memory-budget-gb 12 --no-inline-eval

    # Train BEV with explicit RAM budget
    python train_metadrive.py --modality bev --n-envs 1 --memory-budget-gb 12 --vec-env dummy

    # Train RGB (RunPod 3090, quality-oriented)
    ./run_training.sh train_metadrive.py --modality rgb --n-envs 2 --buffer-size 150000 --batch-size 512 --learning-starts 10000 --train-freq 8 --gradient-steps 4 --learning-rate 1e-4 --vec-env subproc --memory-budget-gb 50 --total-timesteps 1_000_000

    # Run a lightweight CPU debug training pass
    python train_metadrive.py --modality bev --total-timesteps 20_000 --n-envs 1 --device cpu --no-inline-eval

    # Resume interrupted training with explicit config
    python train_metadrive.py --modality bev --n-envs 1 --buffer-size 40000 --vec-env dummy

    # Evaluate a saved model (separate script)
    python evaluate_metadrive.py --model-path ./models/bev/sac_metadrive_final.zip --modality bev --render
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

_headless_patches_applied = False


def _apply_headless_patches() -> None:
    """Apply patches for headless rendering compatibility."""
    global _headless_patches_applied

    if _headless_patches_applied:
        return

    try:
        from metadrive.third_party import simplepbr

        original_init = simplepbr.Pipeline.__init__

        def patched_init(self, *args, **kwargs):
            try:
                original_init(self, *args, **kwargs)
            except AttributeError as e:
                if "set_shader" in str(e):
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
import torch.nn as nn

np.float = np.float64
np.int = np.int64
np.bool = np.bool_

from gymnasium import Env, spaces
from metadrive.envs import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.semantic_camera import SemanticCamera
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

# =============================================================================
# MODALITY CONFIGURATION
# =============================================================================

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

# =============================================================================
# IMAGE MEMORY GUARDRAIL DEFAULTS
# =============================================================================
#
# Image observations (BEV, RGB) store (3, 96, 96) float32 arrays in the replay
# buffer — 108 KB per transition.  With optimize_memory_usage=True (stores obs
# once, not twice) the cost is:
#
#   buffer_size * 108 KB
#
# On a 16 GB desktop with ~7 GB free for the buffer:
#   30 000  transitions → 3.1 GB  ✓
#  100 000  transitions → 10.3 GB ✗  (current default — likely OOM!)
#
# On a 3090 RunPod pod (~48 GB RAM) with ~27 GB free:
#  100 000  transitions → 10.3 GB  ✓
#  150 000  transitions → 15.4 GB  ✓
#
# n_envs is capped at 4 on desktop because each MetaDrive rendering process
# uses ~500 MB of CPU RAM; 8 envs + buffer already saturates 16 GB.
# =============================================================================

IMAGE_DEFAULT_MEMORY_BUDGET_GB = 12.0
IMAGE_BASE_PROCESS_RAM_GB = 4.0
IMAGE_ENV_WORKER_RAM_GB = 1.8
IMAGE_EVAL_ENV_RAM_GB = 1.2
IMAGE_MIN_BUFFER_SIZE = 20_000

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

# =============================================================================
# CUSTOM CNN FEATURE EXTRACTOR FOR IMAGE MODALITIES
# =============================================================================
#
# SB3's default NatureCNN is fine for Atari-style tasks but lacks output
# normalisation, which causes the critic's gradients to destabilize the CNN
# encoder in SAC.  This extractor adds LayerNorm after the linear projection,
# which keeps activations in a well-behaved range and substantially improves
# training stability for BEV and RGB modalities.
#
# Architecture for 96x96 input:
#   Conv(32, 8×8, stride 4) → 23×23
#   Conv(64, 4×4, stride 2) → 10×10
#   Conv(64, 3×3, stride 1) →  8×8
#   Flatten                 → 4096
#   Linear(features_dim)    → 512
#   LayerNorm + ReLU
#
# Used as the image branch inside MetaDriveCombinedExtractor (see below).
# =============================================================================


class _ImageCNN(nn.Module):
    """CNN backbone used by MetaDriveCombinedExtractor."""

    def __init__(self, n_input_channels: int, features_dim: int = 512):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flat size with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, n_input_channels, 96, 96)
            n_flatten = self.cnn(dummy).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            # LayerNorm stabilises the features seen by the actor/critic MLPs.
            # Without it, large-gradient updates from the critic can shift the
            # CNN outputs dramatically and cause divergence.
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(x))


class MetaDriveCombinedExtractor(BaseFeaturesExtractor):
    """Feature extractor for Dict observations with an image + state vector.

    Replaces SB3's default CombinedExtractor for the ``bev`` and ``rgb``
    modalities.  The image branch uses the stabilised _ImageCNN; the state
    branch is a simple flatten.  Both outputs are concatenated before being
    fed into the actor/critic MLPs.

    Args:
        observation_space: Gymnasium Dict space with keys ``"image"`` and
            ``"state"``.
        cnn_output_dim: Dimensionality of the CNN's linear projection layer.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 512,
    ):
        # Compute total feature dim so the base class is happy
        state_dim = int(np.prod(observation_space["state"].shape))
        super().__init__(observation_space, features_dim=cnn_output_dim + state_dim)

        n_input_channels = observation_space["image"].shape[0]  # channel-first
        self.image_extractor = _ImageCNN(n_input_channels, cnn_output_dim)
        self.state_extractor = nn.Flatten()

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        img_features = self.image_extractor(observations["image"])
        state_features = self.state_extractor(observations["state"])
        return torch.cat([img_features, state_features], dim=1)


# =============================================================================
# UINT8 IMAGE REPLAY BUFFER
# =============================================================================
#
# Problem: SB3's DictReplayBuffer stores observations as float32 and hard-
# asserts against optimize_memory_usage=True.  For (3, 96, 96) images this
# means 108 KB per transition, so a 100k buffer needs:
#   2 (obs + next_obs) × 100k × 108 KB = 20.6 GB  ← instant OOM on desktop
#
# Solution: store the image key as uint8 (0–255) and dequantize on sample.
# This gives identical 4× savings without touching the forbidden flag:
#   2 × 100k × 27 KB = 5.15 GB  ← fine on 16 GB
#
# Implementation notes:
#   • We pass the parent a *modified* observation space where the image key
#     has dtype=uint8.  This causes the parent to allocate uint8 arrays
#     directly — no temporary float32 allocation at all.
#   • add()         scales float32 [0,1] → uint8 [0,255] before storing.
#   • _get_samples() scales uint8 [0,255] → float32 [0,1] after loading.
#   • The policy always sees float32; only the buffer internals use uint8.
# =============================================================================


class Uint8ImageDictReplayBuffer(DictReplayBuffer):
    """DictReplayBuffer that stores image observations as uint8 to save 4× RAM.

    Drop-in replacement for DictReplayBuffer.  Pass it via
    ``replay_buffer_class=Uint8ImageDictReplayBuffer`` when constructing SAC.

    Assumptions:
    - The observation space is a ``spaces.Dict`` containing a key ``"image"``
      whose values are float32 in [0, 1].
    - All other keys (e.g. ``"state"``) are stored as-is.
    """

    IMAGE_KEY = "image"

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space,
        device="auto",
        n_envs: int = 1,
        **kwargs,
    ):
        # Build a modified obs space where the image key is uint8.
        # The parent then allocates uint8 arrays directly — no float32
        # temporary allocation occurs.
        uint8_obs_space = self._make_uint8_obs_space(observation_space)
        super().__init__(
            buffer_size,
            uint8_obs_space,
            action_space,
            device=device,
            n_envs=n_envs,
            **kwargs,
        )

    @staticmethod
    def _make_uint8_obs_space(obs_space: spaces.Dict) -> spaces.Dict:
        new_spaces = {}
        for key, space in obs_space.spaces.items():
            if key == Uint8ImageDictReplayBuffer.IMAGE_KEY and isinstance(
                space, spaces.Box
            ):
                new_spaces[key] = spaces.Box(
                    low=0,
                    high=255,
                    shape=space.shape,
                    dtype=np.uint8,
                )
            else:
                new_spaces[key] = space
        return spaces.Dict(new_spaces)

    def add(
        self,
        obs,
        next_obs,
        action,
        reward,
        done,
        infos,
    ):
        """Quantize float32 [0,1] images to uint8 [0,255] before storing."""
        if isinstance(obs, dict) and self.IMAGE_KEY in obs:
            obs = dict(obs)
            obs[self.IMAGE_KEY] = (
                (obs[self.IMAGE_KEY] * 255.0).clip(0, 255).astype(np.uint8)
            )
        if isinstance(next_obs, dict) and self.IMAGE_KEY in next_obs:
            next_obs = dict(next_obs)
            next_obs[self.IMAGE_KEY] = (
                (next_obs[self.IMAGE_KEY] * 255.0).clip(0, 255).astype(np.uint8)
            )
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds, env=None):
        """Dequantize image tensors from uint8 [0,255] back to float32 [0,1]."""
        samples = super()._get_samples(batch_inds, env=env)

        obs = dict(samples.observations)
        next_obs = dict(samples.next_observations)

        if self.IMAGE_KEY in obs:
            obs[self.IMAGE_KEY] = obs[self.IMAGE_KEY].float() / 255.0
        if self.IMAGE_KEY in next_obs:
            next_obs[self.IMAGE_KEY] = next_obs[self.IMAGE_KEY].float() / 255.0

        return DictReplayBufferSamples(
            observations=obs,
            next_observations=next_obs,
            actions=samples.actions,
            dones=samples.dones,
            rewards=samples.rewards,
        )


# =============================================================================
# GYM → GYMNASIUM WRAPPER
# =============================================================================


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

            modality_config = MODALITY_CONFIG.get(
                self.modality, MODALITY_CONFIG["sparse_lidar"]
            )
            if is_observation and modality_config.get("image_observation", False):
                image_size = modality_config.get("image_size", 128)
                stack_size = modality_config.get("stack_size", 1)
                num_channels = 3 * stack_size
                new_shape = (num_channels, image_size, image_size)

                low = np.zeros(new_shape, dtype=np.float32)
                high = np.ones(new_shape, dtype=np.float32)

                return spaces.Box(low=low, high=high, shape=new_shape, dtype=np.float32)

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

        if isinstance(obs, dict) and self.modality in ["rgb", "bev"]:
            obs = {
                "image": self._format_stacked_image(obs["image"]),
                "state": np.asarray(obs["state"], dtype=np.float32),
            }
        else:
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

        if isinstance(obs, dict) and self.modality in ["rgb", "bev"]:
            obs = {
                "image": self._format_stacked_image(obs["image"]),
                "state": np.asarray(obs["state"], dtype=np.float32),
            }
        else:
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
        env_config["image_observation"] = True

        camera_type = config["camera"]
        stack_size = config.get("stack_size", 1)

        if camera_type == "rgb":
            env_config["sensors"] = {"rgb": (RGBCamera, 96, 96)}
        elif camera_type == "topdown":
            env_config["sensors"] = {"topdown": (SemanticCamera, 96, 96)}

        env_config["vehicle_config"] = dict(
            image_source=camera_type,
            show_lidar=False,
            lidar=dict(num_lasers=0, distance=0),
            show_lane_line_detector=False,
            lane_line_detector=dict(num_lasers=0, distance=20),
            show_side_detector=False,
            side_detector=dict(num_lasers=0, distance=50),
        )

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
    _apply_headless_patches()

    config = get_env_config(modality, is_eval=is_eval, render=render)
    env = MetaDriveEnvWrapper(config=config)
    return GymToGymnasiumWrapper(env, modality=modality)


# =============================================================================
# CALLBACKS
# =============================================================================


class GarbageCollectionCallback(BaseCallback):
    """Periodically trigger garbage collection to reduce memory fragmentation."""

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

    Wraps the model's training operations with torch.autocast to use fp16
    for forward passes while keeping gradients in fp32. Reduces GPU memory
    usage by ~40-50% with minimal impact on training stability.
    """

    def __init__(self, enabled: bool = True, verbose: int = 1):
        super().__init__(verbose)
        self._enabled_flag = enabled
        self.enabled = False
        self.scaler = None
        self._original_train = None
        self._step_count = 0

    def _on_training_start(self) -> None:
        if not self._enabled_flag:
            if self.verbose > 0:
                print("[*] Mixed precision disabled by user.")
            return

        if not torch.cuda.is_available():
            print("[!] CUDA not available, mixed precision disabled.")
            self.enabled = False
            return

        self.enabled = True
        self.scaler = torch.amp.GradScaler("cuda")

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


# =============================================================================
# TRAINING
# =============================================================================


def _print_buffer_memory_estimate(buffer_size: int, modality: str) -> None:
    """Print how much RAM the replay buffer will consume (uint8 image storage)."""
    config = MODALITY_CONFIG.get(modality, {})
    if not config.get("image_observation", False):
        return

    img_size = config.get("image_size", 96)
    channels = 3 * config.get("stack_size", 1)
    # uint8 = 1 byte; DictReplayBuffer stores obs AND next_obs (×2)
    bytes_per_obs_uint8 = channels * img_size * img_size * 1
    total_gb = 2 * buffer_size * bytes_per_obs_uint8 / (1024**3)

    print(
        f"[*] Replay buffer estimate (uint8 images): "
        f"{buffer_size:,} transitions × 2 × {bytes_per_obs_uint8 // 1024} KB "
        f"= {total_gb:.2f} GB"
    )


def _estimate_image_replay_ram_gb(buffer_size: int, modality: str) -> float:
    """Estimate replay-buffer RAM usage in GB for image modalities."""
    config = MODALITY_CONFIG.get(modality, {})
    if not config.get("image_observation", False):
        return 0.0

    img_size = config.get("image_size", 96)
    channels = 3 * config.get("stack_size", 1)
    bytes_per_obs_uint8 = channels * img_size * img_size
    return 2 * buffer_size * bytes_per_obs_uint8 / (1024**3)


def _estimate_image_total_ram_gb(
    *,
    buffer_size: int,
    modality: str,
    n_envs: int,
    base_process_ram_gb: float,
    env_worker_ram_gb: float,
    include_inline_eval: bool,
    eval_env_ram_gb: float,
) -> float:
    """Estimate total host RAM for image training (replay + env workers + overhead)."""
    replay_gb = _estimate_image_replay_ram_gb(buffer_size=buffer_size, modality=modality)
    eval_gb = eval_env_ram_gb if include_inline_eval else 0.0
    return replay_gb + base_process_ram_gb + (n_envs * env_worker_ram_gb) + eval_gb


def _fit_image_config_to_memory_budget(
    *,
    modality: str,
    n_envs: int,
    buffer_size: int,
    include_inline_eval: bool,
    base_process_ram_gb: float,
    env_worker_ram_gb: float,
    eval_env_ram_gb: float,
    min_buffer_size: int,
    memory_budget_gb: float,
    strict_memory_budget: bool,
) -> tuple[int, int, float]:
    """Clamp n_envs/buffer_size for image modalities to fit host RAM budget."""
    est = _estimate_image_total_ram_gb(
        buffer_size=buffer_size,
        modality=modality,
        n_envs=n_envs,
        base_process_ram_gb=base_process_ram_gb,
        env_worker_ram_gb=env_worker_ram_gb,
        include_inline_eval=include_inline_eval,
        eval_env_ram_gb=eval_env_ram_gb,
    )

    while est > memory_budget_gb and n_envs > 1:
        n_envs -= 1
        est = _estimate_image_total_ram_gb(
            buffer_size=buffer_size,
            modality=modality,
            n_envs=n_envs,
            base_process_ram_gb=base_process_ram_gb,
            env_worker_ram_gb=env_worker_ram_gb,
            include_inline_eval=include_inline_eval,
            eval_env_ram_gb=eval_env_ram_gb,
        )

    while est > memory_budget_gb and buffer_size > min_buffer_size:
        next_buffer = int(buffer_size * 0.9)
        buffer_size = max(min_buffer_size, next_buffer)
        est = _estimate_image_total_ram_gb(
            buffer_size=buffer_size,
            modality=modality,
            n_envs=n_envs,
            base_process_ram_gb=base_process_ram_gb,
            env_worker_ram_gb=env_worker_ram_gb,
            include_inline_eval=include_inline_eval,
            eval_env_ram_gb=eval_env_ram_gb,
        )

    if est > memory_budget_gb and strict_memory_budget:
        raise ValueError(
            "Image training configuration exceeds memory budget even after adaptation. "
            f"Estimated={est:.2f} GB, budget={memory_budget_gb:.2f} GB. "
            "Increase --memory-budget-gb or lower --n-envs/--buffer-size."
        )

    return n_envs, buffer_size, est


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
    inline_eval: bool = True,
    eval_freq: int = 100_000,
    eval_episodes: int = 8,
    vec_env: str = "auto",
    memory_budget_gb: Optional[float] = None,
    resume_from: Optional[str] = None,
):
    modality_config = MODALITY_CONFIG.get(modality, MODALITY_CONFIG["sparse_lidar"])
    policy_type = modality_config["policy"]
    is_image_modality = modality_config.get("image_observation", False)
    resolved_memory_budget_gb = memory_budget_gb
    resolved_save_replay_buffer = not is_image_modality
    strict_memory_budget = True

    # -------------------------------------------------------------------------
    # IMAGE MODALITY: apply memory-budget guardrail
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    if is_image_modality:
        resolved_memory_budget_gb = (
            float(memory_budget_gb)
            if memory_budget_gb is not None and memory_budget_gb > 0
            else IMAGE_DEFAULT_MEMORY_BUDGET_GB
        )

        if inline_eval:
            est_with_eval = _estimate_image_total_ram_gb(
                buffer_size=buffer_size,
                modality=modality,
                n_envs=n_envs,
                base_process_ram_gb=IMAGE_BASE_PROCESS_RAM_GB,
                env_worker_ram_gb=IMAGE_ENV_WORKER_RAM_GB,
                include_inline_eval=True,
                eval_env_ram_gb=IMAGE_EVAL_ENV_RAM_GB,
            )
            est_without_eval = _estimate_image_total_ram_gb(
                buffer_size=buffer_size,
                modality=modality,
                n_envs=n_envs,
                base_process_ram_gb=IMAGE_BASE_PROCESS_RAM_GB,
                env_worker_ram_gb=IMAGE_ENV_WORKER_RAM_GB,
                include_inline_eval=False,
                eval_env_ram_gb=IMAGE_EVAL_ENV_RAM_GB,
            )
            if (
                est_with_eval > resolved_memory_budget_gb
                and est_without_eval <= resolved_memory_budget_gb
            ):
                inline_eval = False
                print(
                    "[*] Disabled inline eval in image mode to stay within memory budget "
                    f"({resolved_memory_budget_gb:.1f} GB)."
                )

        before_n_envs = n_envs
        before_buffer_size = buffer_size
        n_envs, buffer_size, est_total_gb = _fit_image_config_to_memory_budget(
            modality=modality,
            n_envs=n_envs,
            buffer_size=buffer_size,
            include_inline_eval=inline_eval,
            base_process_ram_gb=IMAGE_BASE_PROCESS_RAM_GB,
            env_worker_ram_gb=IMAGE_ENV_WORKER_RAM_GB,
            eval_env_ram_gb=IMAGE_EVAL_ENV_RAM_GB,
            min_buffer_size=IMAGE_MIN_BUFFER_SIZE,
            memory_budget_gb=resolved_memory_budget_gb,
            strict_memory_budget=strict_memory_budget,
        )
        if n_envs != before_n_envs:
            print(f"[*] n_envs clamped to {n_envs} for memory budget")
        if buffer_size != before_buffer_size:
            print(f"[*] buffer_size clamped to {buffer_size:,} for memory budget")
        print(
            f"[*] Estimated host RAM for image training: {est_total_gb:.2f} GB "
            f"(budget: {resolved_memory_budget_gb:.2f} GB)"
        )

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    save_dir = Path(save_dir) / modality
    save_dir.mkdir(parents=True, exist_ok=True)

    log_dir = save_dir / "logs"
    checkpoint_dir = save_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    csv_log_path = save_dir / "training_log.csv"
    model_path = save_dir / "sac_metadrive_final.zip"
    resume_source = Path(resume_from) if resume_from else model_path

    # -------------------------------------------------------------------------
    # Print config summary
    # -------------------------------------------------------------------------
    print("=" * 70)
    print(f"MetaDrive SAC Training - {modality_config['description']}")
    print("=" * 70)
    print(f"Device:               {device}")
    print(f"Vec env mode:         {vec_env}")
    print(f"Policy:               {policy_type}")
    print(f"Parallel environments:{n_envs}")
    print(
        f"Training seeds:       {TRAIN_SEED_START} – "
        f"{TRAIN_SEED_START + TRAIN_SEED_COUNT - 1}"
    )
    print(f"Total timesteps:      {total_timesteps:,}")
    print(f"Learning starts:      {learning_starts:,}")
    print(f"Buffer size:          {buffer_size:,}")
    print(f"Batch size:           {batch_size}")
    print(f"Train freq:           {train_freq}")
    print(f"Gradient steps:       {gradient_steps}")
    print(f"Learning rate:        {learning_rate}")
    if is_image_modality:
        print(f"CNN extractor:        MetaDriveCombinedExtractor (LayerNorm)")
        print(f"Replay buffer:        Uint8ImageDictReplayBuffer (4× smaller than float32)")
        print(f"Memory budget (RAM):  {resolved_memory_budget_gb:.2f} GB")
        print(f"Strict budget mode:   {'on' if strict_memory_budget else 'off'}")
        print(f"Inline eval mode:     {'on' if inline_eval else 'off'}")
        print(f"Save replay buffer:   {'on' if resolved_save_replay_buffer else 'off'}")
        _print_buffer_memory_estimate(buffer_size, modality)
    print(f"Resume source:        {resume_source}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Create vectorised training environment
    # -------------------------------------------------------------------------
    print("\nCreating training environment...")
    print(f"Creating {n_envs} parallel environments...")

    if vec_env == "dummy":
        vec_env_cls = DummyVecEnv
    elif vec_env == "subproc":
        vec_env_cls = SubprocVecEnv
    else:
        vec_env_cls = DummyVecEnv if (is_image_modality and n_envs <= 2) else SubprocVecEnv

    print(f"Vector env class:     {vec_env_cls.__name__}")

    env = make_vec_env(
        lambda: create_metadrive_env(modality, is_eval=False),
        n_envs=n_envs,
        vec_env_cls=vec_env_cls,
    )

    print(f"\nEnvironment created successfully with {n_envs} parallel workers!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # -------------------------------------------------------------------------
    # Load or create model
    # -------------------------------------------------------------------------
    model_loaded = False
    start_timesteps = 0
    replay_buffer_path = save_dir / "replay_buffer.pkl"

    if resume_source.exists():
        try:
            load_overrides: dict[str, Any] = {
                "learning_rate": learning_rate,
                "buffer_size": buffer_size,
                "learning_starts": learning_starts,
                "batch_size": batch_size,
                "train_freq": train_freq,
                "gradient_steps": gradient_steps,
            }
            if is_image_modality:
                load_overrides["replay_buffer_class"] = Uint8ImageDictReplayBuffer

            model = SAC.load(
                str(resume_source),
                env=env,
                tensorboard_log=str(log_dir),
                device=device,
                custom_objects=load_overrides,
                **load_overrides,
            )
            start_timesteps = model.num_timesteps
            print(f"[+] Resumed from '{resume_source}' at step {start_timesteps:,}")
            print("[*] Effective resumed hyperparameters:")
            print(f"    learning_rate:     {model.learning_rate}")
            print(f"    buffer_size:       {model.buffer_size:,}")
            print(f"    learning_starts:   {model.learning_starts:,}")
            print(f"    batch_size:        {model.batch_size}")
            print(f"    train_freq:        {model.train_freq}")
            print(f"    gradient_steps:    {model.gradient_steps}")
            if is_image_modality:
                print(
                    "    replay_buffer_cls: "
                    f"{model.replay_buffer_class.__name__}"
                )
            if replay_buffer_path.exists():
                replay_file_size_gb = replay_buffer_path.stat().st_size / (1024**3)
                if is_image_modality:
                    if resolved_memory_budget_gb is None:
                        resolved_memory_budget_gb = 12.0
                    max_replay_load_gb = max(2.0, resolved_memory_budget_gb * 0.6)
                    should_load_replay = replay_file_size_gb <= max_replay_load_gb
                    if not should_load_replay:
                        print(
                            "[*] Skipping replay buffer load for image modality "
                            f"(file={replay_file_size_gb:.2f} GB exceeds auto limit "
                            f"{max_replay_load_gb:.2f} GB)."
                        )
                else:
                    should_load_replay = True

                try:
                    if should_load_replay:
                        model.load_replay_buffer(str(replay_buffer_path))
                        print(f"[+] Replay buffer loaded from: {replay_buffer_path}")
                    else:
                        print("[*] Replay buffer not loaded (policy decision).")
                except Exception as rb_e:
                    print(
                        f"[!] Replay buffer load failed ({rb_e}) — using fresh buffer."
                    )
            model_loaded = True
        except Exception as e:
            print(f"[!] Model load failed ({e}) — starting fresh.")
    elif resume_from is not None:
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_source}")

    if not model_loaded:
        print("[>] Initializing new SAC model...")

        # -----------------------------------------------------------------
        # policy_kwargs for image modalities
        # -----------------------------------------------------------------
        # MetaDriveCombinedExtractor replaces SB3's default CombinedExtractor
        # for Dict observations.  The key difference is the _ImageCNN's
        # LayerNorm output, which prevents the critic gradients from driving
        # large shifts in the CNN encoder's feature distribution.
        #
        # net_arch=[256, 256] gives the actor and critic moderate-sized MLPs
        # on top of the 512-dim (CNN) + state concatenation.  The total
        # feature dim fed into the MLP is therefore ~521 (512 + ~9 state).
        # -----------------------------------------------------------------
        policy_kwargs: dict[str, Any] = {}

        if is_image_modality:
            policy_kwargs = {
                "features_extractor_class": MetaDriveCombinedExtractor,
                "features_extractor_kwargs": {"cnn_output_dim": 512},
                "net_arch": [256, 256],
                # Images are already in [0, 1] (normalised in the wrapper)
                "normalize_images": False,
            }
        else:
            # For LiDAR modalities keep the default MLP policy unchanged
            policy_kwargs = {}

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
            # Use Uint8ImageDictReplayBuffer for image modalities.
            # DictReplayBuffer (the SB3 default for Dict obs) hard-asserts
            # against optimize_memory_usage=True, so we solve the memory
            # problem differently: store images as uint8 (1 byte) instead of
            # float32 (4 bytes), giving 4× smaller buffers at no quality cost.
            replay_buffer_class=(
                Uint8ImageDictReplayBuffer if is_image_modality else None
            ),
        )

    remaining_timesteps = max(0, total_timesteps - start_timesteps)
    print(
        f"[>] Training {remaining_timesteps:,} remaining steps (budget {total_timesteps:,})"
    )

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Signal / atexit safety save
    # -------------------------------------------------------------------------
    save_model_called = False

    def save_model_on_exit():
        nonlocal save_model_called
        if save_model_called:
            return
        save_model_called = True
        print("\n[+] Saving model and replay buffer (on exit)...")
        try:
            model.save(str(model_path))
            print(f"[+] Model saved to: {model_path}")
            if resolved_save_replay_buffer:
                model.save_replay_buffer(str(replay_buffer_path))
                print(f"[+] Replay buffer saved to: {replay_buffer_path}")
            else:
                print("[*] Replay buffer save skipped by policy.")
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


# =============================================================================
# ENTRY POINT
# =============================================================================

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
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1_000_000,
        help="Replay buffer size",
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=1000,
        help="Timesteps before learning starts",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="SAC minibatch size",
    )
    parser.add_argument(
        "--train-freq",
        type=int,
        default=1,
        help="Environment steps between update phases",
    )
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=1,
        help="Gradient updates per training phase",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["auto", "cuda", "cpu"],
        help="Device for training",
    )
    parser.add_argument(
        "--no-inline-eval",
        action="store_true",
        default=False,
        help="Disable inline evaluation during training",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        default=False,
        help="Enable AMP (fp16) training for reduced VRAM usage",
    )
    parser.add_argument(
        "--vec-env",
        type=str,
        default="auto",
        choices=["auto", "dummy", "subproc"],
        help="Vector environment backend",
    )
    parser.add_argument(
        "--memory-budget-gb",
        type=float,
        default=None,
        help=(
            "Host RAM budget for image modalities (BEV/RGB). "
            "Defaults to 12GB if omitted."
        ),
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./models",
        help="Base output directory for models/logs (modality subfolder is appended)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help=(
            "Path to a specific checkpoint/model zip to resume from. "
            "If omitted, auto-resume uses <save-dir>/<modality>/sac_metadrive_final.zip"
        ),
    )

    args = parser.parse_args()

    train_sac_agent(
        modality=args.modality,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        n_envs=args.n_envs,
        save_dir=args.save_dir,
        device=args.device,
        mixed_precision=args.mixed_precision,
        inline_eval=not args.no_inline_eval,
        vec_env=args.vec_env,
        memory_budget_gb=args.memory_budget_gb,
        resume_from=args.resume_from,
    )
