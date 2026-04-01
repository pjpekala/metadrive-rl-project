"""Modular CNN feature extractors for different observation types.

This module provides CNN feature extractors for image-based observations
in SAC training. Supports Bird's Eye View (BEV) semantic maps, RGB camera
images, and provides a modular base class for easy extension.

Usage:
    >>> from src.sac_training.feature_extractors import BEVCNN, get_feature_extractor_config
    >>> policy_kwargs = get_feature_extractor_config("bev", features_dim=256)
    >>> print(policy_kwargs)
    {'features_extractor_class': <class 'BEVCNN'>, 'features_extractor_kwargs': {...}}
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ModularCNN(BaseFeaturesExtractor):
    """Base CNN feature extractor for image observations.
    
    This base class provides common functionality for all image-based
    feature extractors including channel permutation (HWC -> CHW) and
    float normalization. Subclasses implement the specific CNN architecture.
    
    Attributes:
        observation_space: Gymnasium Box observation space
        features_dim: Number of output features
        normalized_image: Whether input images are already normalized
    
    Example:
        >>> class MyCNN(ModularCNN):
        ...     def _build_cnn(self, n_input_channels):
        ...         return nn.Sequential(...)
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        """Initialize the modular CNN.
        
        Args:
            observation_space: Gymnasium Box space with shape (H, W, C), (C, H, W), or (H, W, C, frames)
            features_dim: Number of features to output
            normalized_image: Whether images are already normalized to [0, 1]
        """
        super().__init__(observation_space, features_dim)
        self.normalized_image = normalized_image
        
        # Determine input channels from observation space
        self.n_input_channels = self._get_input_channels(observation_space)
        
        # Build the CNN architecture (implemented by subclasses)
        self.cnn = self._build_cnn(self.n_input_channels)
        
        # Compute the flattened size after convolutions
        # Extract actual image dimensions from observation space
        shape = observation_space.shape
        if len(shape) == 3:
            if shape[2] in [1, 3, 5]:
                # Channel-last format: (H, W, C)
                height, width = shape[0], shape[1]
            else:
                # Channel-first format: (C, H, W)
                height, width = shape[1], shape[2]
        elif len(shape) == 4:
            # Frame-stacked observation: (H, W, C, frames)
            height, width = shape[0], shape[1]
        else:
            raise ValueError(f"Unexpected observation shape: {shape}")
        
        with torch.no_grad():
            sample = torch.zeros(1, self.n_input_channels, height, width)
            n_flatten = self.cnn(sample).reshape(1, -1).shape[1]
        
        # Fully connected layer to get features_dim output
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
    
    def _get_input_channels(self, observation_space: spaces.Box) -> int:
        """Determine number of input channels from observation space.
        
        Handles both 3D and 4D observations. For 4D observations (with frame stacking),
        multiplies the channel dimension by the frame stack dimension.
        
        Args:
            observation_space: Gymnasium Box space
            
        Returns:
            Number of input channels
            
        Raises:
            ValueError: If observation space is not 3D or 4D
        """
        shape = observation_space.shape
        
        if len(shape) == 3:
            # Standard 3D observation: (H, W, C) or (C, H, W)
            if shape[2] in [1, 3, 5]:
                # Channel-last format: (H, W, C)
                return shape[2]
            # Channel-first format: (C, H, W)
            return shape[0]
        elif len(shape) == 4:
            # Frame-stacked observation: (H, W, C, frames) or similar
            # Treat the last dimension as additional stacked frames
            if shape[2] in [1, 3, 5]:
                # Channel-last format: (H, W, C, frames)
                return shape[2] * shape[3]
            # Channel-first format or other: multiply channels by frames
            return shape[0] * shape[3]
        else:
            raise ValueError(
                f"Expected 3D or 4D observation space, got shape: {shape}"
            )
    
    def _build_cnn(self, n_input_channels: int) -> nn.Module:
        """Build CNN layers. Override in subclasses.
        
        Args:
            n_input_channels: Number of input channels
            
        Returns:
            CNN module
            
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement _build_cnn()")
    
    def _normalize(self, observations: torch.Tensor) -> torch.Tensor:
        """Normalize observations to [0, 1] if needed.
        
        Args:
            observations: Input observations tensor
            
        Returns:
            Normalized observations
        """
        observations = observations.float()
        if not self.normalized_image:
            # Assume input is in [0, 255] range, normalize to [0, 1]
            if observations.max() > 1.0:
                observations = observations / 255.0
        return observations
    
    def _prepare_channels(self, observations: torch.Tensor) -> torch.Tensor:
        """Prepare observations by ensuring channel-first format.
        
        Handles both standard 4D tensors and frame-stacked 5D tensors.
        
        Args:
            observations: Input observations tensor
            
        Returns:
            Observations in (B, C, H, W) format
        """
        # Ensure batch dimension
        if observations.dim() == 3:
            observations = observations.unsqueeze(0)
        
        # Handle 5D frame-stacked observations: (B, H, W, C, frames)
        # Reshape to (B, H, W, C*frames) then permute to (B, C*frames, H, W)
        if observations.dim() == 5:
            batch, h, w, c, frames = observations.shape
            observations = observations.permute(0, 1, 2, 3, 4).reshape(batch, h, w, c * frames)
        
        # Check if we need to permute from (B, H, W, C) to (B, C, H, W)
        if observations.shape[-1] in [1, 3, 5, 9] and len(observations.shape) == 4:
            observations = observations.permute(0, 3, 1, 2)
        
        return observations
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features from observations.
        
        Args:
            observations: Tensor of shape (batch, C, H, W) or (batch, H, W, C)
            
        Returns:
            Features tensor of shape (batch, features_dim)
        """
        # Prepare channels
        observations = self._prepare_channels(observations)
        
        # Normalize
        observations = self._normalize(observations)
        
        # Apply CNN layers
        cnn_features = self.cnn(observations)
        
        # Apply linear projection
        return self.linear(cnn_features)


class BEVCNN(ModularCNN):
    """CNN feature extractor for Bird's Eye View semantic maps.
    
    Designed for 5-channel semantic segmentation inputs from MetaDrive's
    TopDownMetaDriveEnvV2 with BEV observations. The 5 channels represent:
    - Channel 0: Road surface
    - Channel 1: Lane markings
    - Channel 2: Navigation route
    - Channel 3: Other vehicles
    - Channel 4: Pedestrians
    
    Architecture:
    - Conv1: 5 channels → 32, 8x8 kernel, stride 4, ReLU
    - Conv2: 32 channels → 64, 4x4 kernel, stride 2, ReLU
    - Conv3: 64 channels → 64, 3x3 kernel, stride 1, ReLU
    - Flatten → Linear: 64*7*7=3136 → features_dim, ReLU
    
    Attributes:
        observation_space: Gymnasium Box space
        features_dim: Number of output features
    
    Example:
        >>> from src.sac_training.feature_extractors import BEVCNN
        >>> bev_cnn = BEVCNN(observation_space, features_dim=256)
        >>> features = bev_cnn(observations)
    """
    
    def _build_cnn(self, n_input_channels: int) -> nn.Module:
        """Build BEV-specific CNN architecture.
        
        Args:
            n_input_channels: Number of input channels (typically 5 for BEV)
            
        Returns:
            CNN module
        """
        return nn.Sequential(
            # Layer 1: n_input_channels → 32, aggressive downsampling
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            # Layer 2: 32 → 64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            # Layer 3: 64 → 64
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )


class RGBCNN(ModularCNN):
    """CNN feature extractor for RGB camera observations.
    
    Designed for 3-channel RGB camera inputs from MetaDrive's
    front-facing camera. Uses a similar architecture to BEVCNN
    but adapted for natural images instead of semantic maps.
    
    Architecture:
    - Conv1: 3 channels → 32, 8x8 kernel, stride 4, ReLU
    - Conv2: 32 channels → 64, 4x4 kernel, stride 2, ReLU
    - Conv3: 64 channels → 64, 3x3 kernel, stride 1, ReLU
    - Flatten → Linear: 64*7*7=3136 → features_dim, ReLU
    
    Attributes:
        observation_space: Gymnasium Box space
        features_dim: Number of output features
    
    Example:
        >>> from src.sac_training.feature_extractors import RGBCNN
        >>> rgb_cnn = RGBCNN(observation_space, features_dim=256)
        >>> features = rgb_cnn(observations)
    """
    
    def _build_cnn(self, n_input_channels: int) -> nn.Module:
        """Build RGB-specific CNN architecture.
        
        Args:
            n_input_channels: Number of input channels (typically 3 for RGB)
            
        Returns:
            CNN module
        """
        return nn.Sequential(
            # Layer 1: n_input_channels → 32, aggressive downsampling
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            # Layer 2: 32 → 64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            # Layer 3: 64 → 64
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )


def get_feature_extractor_config(
    sensor_type: str,
    features_dim: int = 256,
    normalized_image: bool = False,
) -> dict:
    """Get policy kwargs for using feature extractors with SAC.
    
    Returns the appropriate policy_kwargs dictionary for SAC based on
    the sensor type. Image-based sensors (BEV, RGB) use custom CNN feature
    extractors, while LiDAR-based sensors use standard MLP policies.
    
    Args:
        sensor_type: Type of sensor ("lidar", "bev", "rgb", "dense-lidar")
        features_dim: Number of features to extract (default: 256)
        normalized_image: Whether images are already normalized (default: False)
        
    Returns:
        Dictionary to pass as policy_kwargs to SAC. Empty dict for LiDAR sensors,
        dict with features_extractor_class and features_extractor_kwargs for image sensors.
        
    Raises:
        ValueError: If sensor_type is not supported
        
    Example:
        >>> from src.sac_training.feature_extractors import get_feature_extractor_config
        >>> # Get config for BEV training
        >>> policy_kwargs = get_feature_extractor_config("bev", features_dim=256)
        >>> print(policy_kwargs)
        {'features_extractor_class': <class 'BEVCNN'>, ...}
        
        >>> # Get config for LiDAR training (no custom extractor needed)
        >>> policy_kwargs = get_feature_extractor_config("lidar")
        >>> print(policy_kwargs)
        {}
        
        >>> # Use with SAC
        >>> from stable_baselines3 import SAC
        >>> model = SAC(
        ...     "CnnPolicy",
        ...     env,
        ...     policy_kwargs=get_feature_extractor_config("bev", 256),
        ...     ...
        ... )
    """
    if sensor_type == "bev":
        return {
            "features_extractor_class": BEVCNN,
            "features_extractor_kwargs": {
                "features_dim": features_dim,
                "normalized_image": normalized_image,
            },
        }
    elif sensor_type == "rgb":
        return {
            "features_extractor_class": RGBCNN,
            "features_extractor_kwargs": {
                "features_dim": features_dim,
                "normalized_image": normalized_image,
            },
        }
    elif sensor_type in ("lidar", "dense-lidar"):
        # LiDAR uses MlpPolicy, no custom feature extractor needed
        return {}
    else:
        raise ValueError(
            f"Unsupported sensor type: '{sensor_type}'. "
            f"Supported types: 'lidar', 'bev', 'rgb', 'dense-lidar'"
        )


def is_image_sensor(sensor_type: str) -> bool:
    """Check if sensor type produces image observations.
    
    Args:
        sensor_type: Type of sensor
        
    Returns:
        True if sensor produces image observations (needs CnnPolicy)
        
    Example:
        >>> is_image_sensor("bev")
        True
        
        >>> is_image_sensor("lidar")
        False
    """
    return sensor_type in ("bev", "rgb")
