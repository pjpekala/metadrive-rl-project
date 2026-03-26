"""
Custom CNN Feature Extractor for Bird's-Eye View (BEV) Semantic Maps

This module provides a CNN specifically designed for MetaDrive's BEV sensor output,
which consists of 5 semantic channels:
- Channel 0: Road surface
- Channel 1: Lane markings  
- Channel 2: Navigation route
- Channel 3: Other vehicles
- Channel 4: Pedestrians

The architecture is optimized for semantic segmentation maps rather than natural images.
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class BEVCNN(BaseFeaturesExtractor):
    """
    CNN feature extractor for Bird's-Eye View semantic maps.
    
    Designed for 5-channel semantic segmentation inputs from MetaDrive's
    TopDownMetaDriveEnvV2 with BEV observations.
    
    Architecture:
    - Conv1: 5 channels → 32, 8x8 kernel, stride 4, ReLU
    - Conv2: 32 channels → 64, 4x4 kernel, stride 2, ReLU
    - Conv3: 64 channels → 64, 3x3 kernel, stride 1, ReLU
    - Flatten → Linear: 64*7*7=3136 → features_dim (256), ReLU
    
    Args:
        observation_space: Gymnasium Box space with shape (H, W, 5) or (5, H, W)
        features_dim: Number of features to output (default: 512)
        normalized_image: Whether images are already normalized to [0, 1]
    
    Note:
        Unlike NatureCNN which expects 1 or 3 channels (grayscale/RGB),
        this CNN handles arbitrary channel counts, making it suitable for
        semantic maps with multiple channels.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        # Determine input channels from observation space
        # Observation can be (C, H, W) or (H, W, C)
        if len(observation_space.shape) == 3:
            if observation_space.shape[2] in [1, 3, 5]:
                # Channel-last format: (H, W, C)
                n_input_channels = observation_space.shape[2]
            else:
                # Channel-first format: (C, H, W)
                n_input_channels = observation_space.shape[0]
        else:
            raise ValueError(
                f"Expected 3D observation space, got shape: {observation_space.shape}"
            )

        super().__init__(observation_space, features_dim)

        # CNN layers
        self.cnn = nn.Sequential(
            # Layer 1: 5 channels → 32, aggressive downsampling
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            # Layer 2: 32 → 64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            # Layer 3: 64 → 64
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # Flatten will be applied in forward
        )

        # Compute the flattened size after convolutions
        # For 84x84 input: 84→19→8→6 (after 3 conv layers)
        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, 84, 84)
            n_flatten = self.cnn(sample).reshape(1, -1).shape[1]

        # Fully connected layer to get features_dim output
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from BEV observations.
        
        Args:
            observations: Tensor of shape (batch, C, H, W) or (batch, H, W, C)
            
        Returns:
            Features tensor of shape (batch, features_dim)
        """
        # Ensure channel-first format (B, C, H, W)
        if observations.dim() == 3:
            # Single observation without batch dimension
            observations = observations.unsqueeze(0)

        # Check if we need to permute from (B, H, W, C) to (B, C, H, W)
        if observations.shape[-1] in [1, 3, 5] and len(observations.shape) == 4:
            # Channel-last format detected
            observations = observations.permute(0, 3, 1, 2)

        # Normalize to [0, 1] if not already normalized
        observations = observations.float()

        # Apply CNN layers
        cnn_features = self.cnn(observations)
        
        # Apply linear projection
        return self.linear(cnn_features)


def get_bev_cnn_config(features_dim: int = 256) -> dict:
    """
    Get policy kwargs for using BEVCNN with SAC.

    Args:
        features_dim: Number of features to extract (default: 256 for memory efficiency)

    Returns:
        Dictionary to pass as policy_kwargs to SAC

    Example:
        >>> from src.sac_training.bev_cnn import get_bev_cnn_config
        >>>
        >>> model = SAC(
        ...     "CnnPolicy",
        ...     env,
        ...     policy_kwargs=get_bev_cnn_config(features_dim=256),
        ...     ...
        ... )
    """
    return {
        "features_extractor_class": BEVCNN,
        "features_extractor_kwargs": {
            "features_dim": features_dim,
            "normalized_image": False,  # MetaDrive already normalizes
        },
    }
