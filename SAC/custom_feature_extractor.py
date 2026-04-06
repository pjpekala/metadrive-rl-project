"""
Custom Feature Extractors for Vision-Based RL in MetaDrive
==========================================================

This module provides a custom feature extractor that processes both image
observations (from BEV or RGB cameras) and state vectors (velocity, steering, etc.)
using a multi-branch architecture.

Architecture:
    Input: {"image": (3, H, W), "state": (state_dim,)}

    CNN Branch (for images):
        Conv2d(3, 32, 8, stride=4) -> ReLU
        Conv2d(32, 64, 4, stride=2) -> ReLU
        Conv2d(64, 128, 3, stride=1) -> ReLU
        Conv2d(128, 128, 3, stride=1) -> ReLU
        Flatten -> Linear(*, cnn_features_dim) -> ReLU

    MLP Branch (for state):
        Linear(state_dim, 128) -> ReLU
        Linear(128, 128) -> ReLU

    Fusion:
        Concatenate(cnn_features, state_features)
        Linear(cnn_features_dim + 128, features_dim) -> ReLU

    Output: (features_dim,) tensor

Key improvements over default NatureCNN:
    1. Separate processing of image and state information
    2. Additional conv layer for better feature extraction at 128x128
    3. Learned fusion of visual and state features
    4. State vector provides velocity/motion info (replaces frame stacking)
"""

from typing import Dict, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CombinedImageStateFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for Dict observations containing both image and state.

    This extractor is designed for autonomous driving tasks where:
    - Images provide spatial/visual information (road, lanes, vehicles)
    - State vector provides ego-vehicle information (velocity, steering, heading)

    The architecture processes each modality separately then fuses them,
    allowing the network to learn the optimal combination of visual and
    state features for decision making.

    Args:
        observation_space: Dict space with 'image' and 'state' keys
        features_dim: Output dimension of the combined features (default: 512)
        cnn_output_dim: Output dimension of CNN branch before fusion (default: 512)
        state_output_dim: Output dimension of state branch before fusion (default: 128)

    Expected observation_space structure:
        Dict({
            "image": Box(low=0, high=1, shape=(C, H, W), dtype=float32),
            "state": Box(low=-inf, high=inf, shape=(state_dim,), dtype=float32)
        })
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 512,
        cnn_output_dim: int = 512,
        state_output_dim: int = 128,
    ):
        # Initialize base class with the combined features dimension
        super().__init__(observation_space, features_dim)

        # Extract spaces
        assert isinstance(observation_space, spaces.Dict), (
            f"Expected Dict observation space, got {type(observation_space)}"
        )
        assert "image" in observation_space.spaces, (
            "Observation space must contain 'image' key"
        )
        assert "state" in observation_space.spaces, (
            "Observation space must contain 'state' key"
        )

        image_space = observation_space.spaces["image"]
        state_space = observation_space.spaces["state"]

        # Get image dimensions (expected: C, H, W)
        assert len(image_space.shape) == 3, (
            f"Expected 3D image (C, H, W), got shape {image_space.shape}"
        )
        n_input_channels = image_space.shape[0]
        image_height = image_space.shape[1]
        image_width = image_space.shape[2]

        # Get state dimension
        assert len(state_space.shape) == 1, (
            f"Expected 1D state vector, got shape {state_space.shape}"
        )
        state_dim = state_space.shape[0]

        # Store dimensions for later use
        self.cnn_output_dim = cnn_output_dim
        self.state_output_dim = state_output_dim

        # =====================================================================
        # CNN Branch for Image Processing
        # =====================================================================
        # Enhanced NatureCNN with additional layer for 128x128 inputs
        # Architecture designed to progressively reduce spatial dimensions
        # while increasing feature channels

        self.cnn = nn.Sequential(
            # Layer 1: (C, 128, 128) -> (32, 31, 31)
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            # Layer 2: (32, 31, 31) -> (64, 14, 14)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            # Layer 3: (64, 14, 14) -> (128, 12, 12)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # Layer 4: (128, 12, 12) -> (128, 10, 10) - NEW layer for 128x128
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate CNN output size by doing a forward pass
        with torch.no_grad():
            sample_image = torch.zeros(1, n_input_channels, image_height, image_width)
            cnn_flatten_dim = self.cnn(sample_image).shape[1]

        # Linear layer to reduce CNN features to desired dimension
        self.cnn_linear = nn.Sequential(
            nn.Linear(cnn_flatten_dim, cnn_output_dim),
            nn.ReLU(),
        )

        # =====================================================================
        # MLP Branch for State Processing
        # =====================================================================
        # Simple 2-layer MLP to process state vector
        # State typically contains: velocity, steering, heading, navigation info

        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, state_output_dim),
            nn.ReLU(),
        )

        # =====================================================================
        # Fusion Layer
        # =====================================================================
        # Combine CNN and state features into final representation

        fusion_input_dim = cnn_output_dim + state_output_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, features_dim),
            nn.ReLU(),
        )

        # Print architecture summary
        print(f"[CombinedImageStateFeaturesExtractor] Architecture:")
        print(f"  Image input: ({n_input_channels}, {image_height}, {image_width})")
        print(f"  State input: ({state_dim},)")
        print(f"  CNN flatten: {cnn_flatten_dim} -> {cnn_output_dim}")
        print(f"  State MLP: {state_dim} -> {state_output_dim}")
        print(f"  Fusion: {fusion_input_dim} -> {features_dim}")
        print(f"  Total output: {features_dim} features")

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process observations through CNN and MLP branches, then fuse.

        Args:
            observations: Dict with 'image' and 'state' tensors
                - image: (batch_size, C, H, W) tensor
                - state: (batch_size, state_dim) tensor

        Returns:
            features: (batch_size, features_dim) tensor
        """
        # Extract image and state from observations dict
        image = observations["image"]
        state = observations["state"]

        # Process image through CNN branch
        cnn_features = self.cnn(image)
        cnn_features = self.cnn_linear(cnn_features)

        # Process state through MLP branch
        state_features = self.state_mlp(state)

        # Concatenate and fuse
        combined = torch.cat([cnn_features, state_features], dim=1)
        features = self.fusion(combined)

        return features


class ImageOnlyFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for image-only observations (no state vector).

    This is a fallback extractor for cases where state vector is not available
    or when testing pure vision-based approaches.

    Uses the same enhanced CNN architecture as CombinedImageStateFeaturesExtractor
    but without the state branch.

    Args:
        observation_space: Box space with shape (C, H, W)
        features_dim: Output dimension (default: 512)
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 512,
    ):
        super().__init__(observation_space, features_dim)

        assert isinstance(observation_space, spaces.Box), (
            f"Expected Box observation space, got {type(observation_space)}"
        )
        assert len(observation_space.shape) == 3, (
            f"Expected 3D image (C, H, W), got shape {observation_space.shape}"
        )

        n_input_channels = observation_space.shape[0]
        image_height = observation_space.shape[1]
        image_width = observation_space.shape[2]

        # Same CNN architecture as combined extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate output size
        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, image_height, image_width)
            cnn_flatten_dim = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(cnn_flatten_dim, features_dim),
            nn.ReLU(),
        )

        print(f"[ImageOnlyFeaturesExtractor] Architecture:")
        print(f"  Image input: ({n_input_channels}, {image_height}, {image_width})")
        print(f"  CNN flatten: {cnn_flatten_dim} -> {features_dim}")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Process image through CNN."""
        return self.linear(self.cnn(observations))


def create_policy_kwargs(
    modality: str,
    use_state: bool = True,
    features_dim: int = 512,
) -> dict:
    """
    Create policy_kwargs dictionary for SAC model initialization.

    Args:
        modality: "bev" or "rgb"
        use_state: Whether to use state vector (True) or image only (False)
        features_dim: Output dimension of feature extractor

    Returns:
        Dictionary to pass as policy_kwargs to SAC
    """
    if use_state:
        return {
            "features_extractor_class": CombinedImageStateFeaturesExtractor,
            "features_extractor_kwargs": {
                "features_dim": features_dim,
                "cnn_output_dim": 512,
                "state_output_dim": 128,
            },
            "normalize_images": False,  # We normalize in the wrapper
        }
    else:
        return {
            "features_extractor_class": ImageOnlyFeaturesExtractor,
            "features_extractor_kwargs": {
                "features_dim": features_dim,
            },
            "normalize_images": False,
        }


# =============================================================================
# Unit Tests
# =============================================================================


def test_combined_extractor():
    """Test the CombinedImageStateFeaturesExtractor."""
    print("\n" + "=" * 60)
    print("Testing CombinedImageStateFeaturesExtractor")
    print("=" * 60)

    # Create observation space
    obs_space = spaces.Dict(
        {
            "image": spaces.Box(low=0, high=1, shape=(3, 128, 128), dtype=np.float32),
            "state": spaces.Box(
                low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
            ),
        }
    )

    # Create extractor
    extractor = CombinedImageStateFeaturesExtractor(obs_space, features_dim=512)

    # Test forward pass
    batch_size = 4
    sample_obs = {
        "image": torch.randn(batch_size, 3, 128, 128),
        "state": torch.randn(batch_size, 19),
    }

    output = extractor(sample_obs)
    print(f"\nForward pass:")
    print(f"  Input image: {sample_obs['image'].shape}")
    print(f"  Input state: {sample_obs['state'].shape}")
    print(f"  Output: {output.shape}")

    assert output.shape == (batch_size, 512), (
        f"Expected output shape ({batch_size}, 512), got {output.shape}"
    )

    # Test backward pass
    loss = output.sum()
    loss.backward()
    print(f"  Backward pass: OK")

    # Count parameters
    total_params = sum(p.numel() for p in extractor.parameters())
    trainable_params = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    print("\n[PASS] CombinedImageStateFeaturesExtractor test passed!")
    return True


def test_image_only_extractor():
    """Test the ImageOnlyFeaturesExtractor."""
    print("\n" + "=" * 60)
    print("Testing ImageOnlyFeaturesExtractor")
    print("=" * 60)

    # Create observation space
    obs_space = spaces.Box(low=0, high=1, shape=(3, 128, 128), dtype=np.float32)

    # Create extractor
    extractor = ImageOnlyFeaturesExtractor(obs_space, features_dim=512)

    # Test forward pass
    batch_size = 4
    sample_obs = torch.randn(batch_size, 3, 128, 128)

    output = extractor(sample_obs)
    print(f"\nForward pass:")
    print(f"  Input: {sample_obs.shape}")
    print(f"  Output: {output.shape}")

    assert output.shape == (batch_size, 512), (
        f"Expected output shape ({batch_size}, 512), got {output.shape}"
    )

    # Test backward pass
    loss = output.sum()
    loss.backward()
    print(f"  Backward pass: OK")

    print("\n[PASS] ImageOnlyFeaturesExtractor test passed!")
    return True


if __name__ == "__main__":
    test_combined_extractor()
    test_image_only_extractor()
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
