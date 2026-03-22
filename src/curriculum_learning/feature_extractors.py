"""
Custom feature extractors for sensor experiments.

Provides CNN architectures for BEV and RGB camera observations that are
compatible with MetaDrive's normalized float32 image format.
"""

from typing import Union

import torch
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for MetaDrive image observations.

    Handles channel-last images (H, W, C) from MetaDrive and converts to
    channel-first format (C, H, W) for convolution operations.

    Architecture: Conv2d -> ReLU -> Conv2d -> ReLU -> Conv2d -> ReLU -> Flatten -> FC
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[-1]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.zeros(1, 84, 84, n_input_channels).permute(0, 3, 1, 2)
            n_flatten = self.cnn(sample).shape[1]
        self.fc = nn.Linear(n_flatten, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.permute(0, 3, 1, 2)
        x = self.cnn(x)
        return self.fc(x)


class RGBCombinedExtractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for RGB camera with Dict observation space.

    Processes image observations with CNN and state vectors with MLP,
    then concatenates the outputs.
    """

    def __init__(self, observation_space: spaces.Dict, cnn_output_dim: int = 256):
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if isinstance(subspace, spaces.Box) and len(subspace.shape) >= 3:
                n_channels = subspace.shape[-1]
                extractors[key] = self._create_image_encoder(n_channels, cnn_output_dim)
                total_concat_size += cnn_output_dim
            else:
                extractors[key] = nn.Flatten()
                total_concat_size += (
                    subspace.shape[-1] if len(subspace.shape) > 0 else subspace.n
                )

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def _create_image_encoder(self, n_channels: int, output_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, output_dim),
        )

    def forward(self, observations: Union[dict, torch.Tensor]) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            if isinstance(observations, dict):
                encoded_tensor_list.append(
                    extractor(observations[key].permute(0, 3, 1, 2))
                )
            else:
                encoded_tensor_list.append(extractor(observations.permute(0, 3, 1, 2)))
        return torch.cat(encoded_tensor_list, dim=1)
