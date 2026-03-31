"""Custom CNN Feature Extractor for Bird's-Eye View (BEV) Semantic Maps.

This module is maintained for backward compatibility. It re-exports
from `src.sac_training.feature_extractors`.

.. deprecated:: 
    This module is maintained for backward compatibility.
    New code should import directly from `src.sac_training.feature_extractors`.

Usage:
    >>> from src.sac_training.bev_cnn import BEVCNN, get_bev_cnn_config
    >>> # Or preferably:
    >>> from src.sac_training.feature_extractors import BEVCNN, get_feature_extractor_config
"""

import warnings

from src.sac_training.feature_extractors import (
    ModularCNN,
    BEVCNN,
    RGBCNN,
    get_feature_extractor_config,
)


def get_bev_cnn_config(features_dim: int = 256) -> dict:
    """Get policy kwargs for using BEVCNN with SAC.
    
    This function is maintained for backward compatibility.
    Consider using `get_feature_extractor_config("bev", features_dim)` instead.
    
    Args:
        features_dim: Number of features to extract (default: 256 for memory efficiency)
        
    Returns:
        Dictionary to pass as policy_kwargs to SAC
    """
    warnings.warn(
        "get_bev_cnn_config() is deprecated. "
        "Use get_feature_extractor_config('bev', features_dim) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_feature_extractor_config("bev", features_dim)


__all__ = [
    "ModularCNN",
    "BEVCNN",
    "RGBCNN",
    "get_bev_cnn_config",
    "get_feature_extractor_config",
]
