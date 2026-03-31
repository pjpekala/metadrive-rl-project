"""SAC Training module for MetaDrive.

This package provides modular SAC training with support for multiple sensor types.

Modules:
- config: Configuration dataclasses for sensors and training
- sensors: Sensor factory for creating environment configurations
- env_factory: Environment factory for creating MetaDrive environments
- feature_extractors: CNN feature extractors for image observations
- callbacks: Training callbacks for visualization and video recording
- train: Main training script

Quick start:
    >>> from src.sac_training.train import SACTrainer
    >>> trainer = SACTrainer(sensor_type="bev")
    >>> model = trainer.train()
"""

from src.sac_training.config import (
    SensorConfig,
    LiDARConfig,
    BEVConfig,
    RGBConfig,
    DenseLiDARConfig,
    TrainingConfig,
    config_to_dict,
    config_from_dict,
)

from src.sac_training.sensors import (
    SensorFactory,
    get_sensor_config,
    is_image_observation,
    get_policy_type,
)

from src.sac_training.env_factory import (
    EnvironmentFactory,
    create_env,
    MetaDriveEnvWrapper,
    TopDownMetaDriveEnvWrapper,
)

from src.sac_training.feature_extractors import (
    ModularCNN,
    BEVCNN,
    RGBCNN,
    get_feature_extractor_config,
    is_image_sensor,
)

from src.sac_training.callbacks import (
    VideoRecorderCallback,
    VisualizationCallback,
)

from src.sac_training.train import (
    SACTrainer,
    test_trained_agent,
)

__all__ = [
    # Config
    "SensorConfig",
    "LiDARConfig",
    "BEVConfig",
    "RGBConfig",
    "DenseLiDARConfig",
    "TrainingConfig",
    "config_to_dict",
    "config_from_dict",
    # Sensors
    "SensorFactory",
    "get_sensor_config",
    "is_image_observation",
    "get_policy_type",
    # Environment
    "EnvironmentFactory",
    "create_env",
    "MetaDriveEnvWrapper",
    "TopDownMetaDriveEnvWrapper",
    # Feature Extractors
    "ModularCNN",
    "BEVCNN",
    "RGBCNN",
    "get_feature_extractor_config",
    "is_image_sensor",
    # Callbacks
    "VideoRecorderCallback",
    "VisualizationCallback",
    # Training
    "SACTrainer",
    "test_trained_agent",
]