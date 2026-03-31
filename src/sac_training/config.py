"""Configuration dataclasses for SAC training with different sensor types.

This module provides type-safe configuration classes for:
- Sensor-specific settings (LiDAR, BEV, RGB camera, Dense LiDAR)
- Training hyperparameters
- Environment settings

Usage:
    >>> from src.sac_training.config import BEVConfig, TrainingConfig
    >>> bev_config = BEVConfig(resolution_size=84, frame_stack=3)
    >>> training_config = TrainingConfig(total_timesteps=1_000_000)
"""

from abc import ABC
from dataclasses import dataclass, asdict, field
from typing import Optional, Any


@dataclass
class SensorConfig(ABC):
    """Base configuration class for all sensor types.
    
    This is an abstract base class that defines the common interface
    for all sensor configurations. Specific sensor types should
    inherit from this class and implement their specific parameters.
    
    Attributes:
        name: Human-readable name for the sensor configuration
        observation_type: Type identifier for the observation mode
    """
    
    name: str = "base_sensor"
    observation_type: str = "base"


@dataclass
class LiDARConfig(SensorConfig):
    """Configuration for default LiDAR sensor observations.
    
    Uses MetaDrive's default LiDAR sensor with configurable laser count
    and range. The LiDAR provides 240 laser beams by default, covering
    a 360-degree field of view.
    
    Attributes:
        lidar_num_lasers: Number of laser beams (default: 240)
        lidar_range: Maximum detection range in meters (default: 50.0)
    
    Example:
        >>> config = LiDARConfig(lidar_num_lasers=240, lidar_range=50.0)
        >>> print(config.lidar_num_lasers)
        240
    """
    
    name: str = "lidar"
    observation_type: str = "lidar"
    lidar_num_lasers: int = 240
    lidar_range: float = 50.0
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.lidar_num_lasers <= 0:
            raise ValueError(f"lidar_num_lasers must be positive, got {self.lidar_num_lasers}")
        if self.lidar_range <= 0:
            raise ValueError(f"lidar_range must be positive, got {self.lidar_range}")


@dataclass
class BEVConfig(SensorConfig):
    """Configuration for Bird's Eye View (BEV) image observations.
    
    Uses MetaDrive's TopDownMetaDriveEnvV2 with semantic segmentation
    maps. The BEV sensor provides 5-channel semantic images showing
    road surface, lane markings, navigation route, other vehicles,
    and pedestrians.
    
    Attributes:
        frame_skip: Number of frames to skip between observations (default: 5)
        frame_stack: Number of frames to stack (default: 3)
        post_stack: Number of additional frames to stack (default: 5)
        norm_pixel: Whether to normalize pixel values (default: True)
        resolution_size: Image resolution in pixels (default: 84)
        distance: View distance in meters (default: 30)
    
    Example:
        >>> config = BEVConfig(resolution_size=84, frame_stack=3)
        >>> print(config.resolution_size)
        84
    """
    
    name: str = "bev"
    observation_type: str = "bev"
    frame_skip: int = 5
    frame_stack: int = 3
    post_stack: int = 5
    norm_pixel: bool = True
    resolution_size: int = 84
    distance: int = 30
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.resolution_size <= 0:
            raise ValueError(f"resolution_size must be positive, got {self.resolution_size}")
        if self.frame_stack <= 0:
            raise ValueError(f"frame_stack must be positive, got {self.frame_stack}")
        if self.frame_skip <= 0:
            raise ValueError(f"frame_skip must be positive, got {self.frame_skip}")
        if self.post_stack < 0:
            raise ValueError(f"post_stack must be non-negative, got {self.post_stack}")


@dataclass
class RGBConfig(SensorConfig):
    """Configuration for RGB camera observations.
    
    Uses MetaDrive's front-facing camera to capture RGB images
    of the driving scene. Suitable for training agents that learn
    directly from visual input like human drivers.
    
    Attributes:
        resolution_size: Image resolution in pixels (default: 84)
        use_front_camera: Enable front camera observation (default: True)
        frame_stack: Number of frames to stack (default: 3)
    
    Example:
        >>> config = RGBConfig(resolution_size=224, frame_stack=3)
        >>> print(config.resolution_size)
        224
    """
    
    name: str = "rgb"
    observation_type: str = "rgb"
    resolution_size: int = 84
    use_front_camera: bool = True
    frame_stack: int = 3
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.resolution_size <= 0:
            raise ValueError(f"resolution_size must be positive, got {self.resolution_size}")
        if self.frame_stack <= 0:
            raise ValueError(f"frame_stack must be positive, got {self.frame_stack}")


@dataclass
class DenseLiDARConfig(SensorConfig):
    """Configuration for dense LiDAR sensor observations.
    
    Uses MetaDrive's LiDAR with significantly increased laser count
    (720 beams) for higher resolution point cloud observations.
    Provides 3x denser coverage than standard LiDAR.
    
    Attributes:
        lidar_num_lasers: Number of laser beams (default: 720)
        lidar_range: Maximum detection range in meters (default: 50.0)
    
    Example:
        >>> config = DenseLiDARConfig(lidar_num_lasers=720)
        >>> print(config.lidar_num_lasers)
        720
    """
    
    name: str = "dense_lidar"
    observation_type: str = "dense_lidar"
    lidar_num_lasers: int = 720
    lidar_range: float = 50.0
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.lidar_num_lasers <= 0:
            raise ValueError(f"lidar_num_lasers must be positive, got {self.lidar_num_lasers}")
        if self.lidar_range <= 0:
            raise ValueError(f"lidar_range must be positive, got {self.lidar_range}")


@dataclass
class TrainingConfig:
    """Configuration for SAC training hyperparameters.
    
    This class defines all hyperparameters for training SAC agents
    on MetaDrive environments. Uses sensible defaults based on
    stable-baselines3 SAC recommendations.
    
    Attributes:
        total_timesteps: Total number of training timesteps (default: 1_000_000)
        learning_rate: Learning rate for Adam optimizer (default: 3e-4)
        buffer_size: Replay buffer size (default: 1_000_000)
        learning_starts: Timesteps before training starts (default: 1000)
        batch_size: Minibatch size for training (default: 256)
        n_envs: Number of parallel environments (default: 8)
        save_dir: Directory to save models (default: "models/sac_metadrive")
        save_freq: Frequency of checkpoint saves (default: 120_000)
        tau: Soft update coefficient for target network (default: 0.005)
        gamma: Discount factor (default: 0.99)
        train_freq: Training frequency (default: 1)
        gradient_steps: Gradient steps per training iteration (default: 1)
        ent_coef: Entropy coefficient, 'auto' for automatic tuning (default: 'auto')
        device: Device to use ('auto', 'cuda', 'cpu', 'mps') (default: 'auto')
        verbose: Verbosity level (default: 1)
    
    Example:
        >>> config = TrainingConfig(total_timesteps=500_000, n_envs=4)
        >>> print(config.total_timesteps)
        500000
    """
    
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    learning_starts: int = 1000
    batch_size: int = 256
    n_envs: int = 8
    save_dir: str = "models/sac_metadrive"
    save_freq: int = 120_000
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = "auto"
    device: str = "auto"
    verbose: int = 1
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.total_timesteps <= 0:
            raise ValueError(f"total_timesteps must be positive, got {self.total_timesteps}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {self.buffer_size}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.n_envs <= 0:
            raise ValueError(f"n_envs must be positive, got {self.n_envs}")
        if self.tau < 0 or self.tau > 1:
            raise ValueError(f"tau must be in [0, 1], got {self.tau}")
        if self.gamma < 0 or self.gamma > 1:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")


def config_to_dict(config: Any) -> dict:
    """Convert a configuration dataclass to a dictionary.
    
    Args:
        config: Configuration dataclass instance
        
    Returns:
        Dictionary representation of the configuration
        
    Example:
        >>> config = LiDARConfig(lidar_num_lasers=240)
        >>> config_to_dict(config)
        {'name': 'lidar', 'observation_type': 'lidar', 'lidar_num_lasers': 240, ...}
    """
    return asdict(config)


def config_from_dict(config_dict: dict, config_class: type) -> Any:
    """Create a configuration instance from a dictionary.
    
    Args:
        config_dict: Dictionary containing configuration values
        config_class: The configuration class to instantiate
        
    Returns:
        Configuration instance
        
    Example:
        >>> config_dict = {'lidar_num_lasers': 720, 'lidar_range': 100.0}
        >>> config = config_from_dict(config_dict, DenseLiDARConfig)
    """
    # Filter only valid fields for the class
    valid_fields = {f.name for f in config_class.__dataclass_fields__.values()}
    filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
    return config_class(**filtered_dict)
