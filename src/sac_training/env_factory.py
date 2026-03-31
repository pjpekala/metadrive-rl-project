"""Environment factory for creating observation-specific MetaDrive environments.

This module provides a unified interface for creating MetaDrive environments
with different sensor types, handling the specific requirements for each
sensor (LiDAR, BEV, RGB, Dense LiDAR).

Usage:
    >>> from src.sac_training.env_factory import EnvironmentFactory
    >>> env = EnvironmentFactory.create_env("bev")
    >>> obs, info = env.reset()
"""

from typing import Optional, Any

from metadrive.envs import MetaDriveEnv
from metadrive.envs.top_down_env import TopDownMetaDriveEnvV2
from stable_baselines3.common.monitor import Monitor

from src.sac_training.sensors import SensorFactory, get_sensor_config


class MetaDriveEnvWrapper(MetaDriveEnv):
    """Wrapper to handle seed modulo operation for parallel environments.
    
    This wrapper ensures that when multiple parallel environments are created,
    each gets a different scenario by wrapping the seed to the valid range.
    
    Attributes:
        config: MetaDrive configuration dictionary
    
    Example:
        >>> config = {"start_seed": 1000, "num_scenarios": 100}
        >>> env = MetaDriveEnvWrapper(config=config)
        >>> obs, info = env.reset(seed=1500)  # Will use seed 1000 + (1500 % 100)
    """
    
    def __init__(self, config: dict) -> None:
        """Initialize the wrapper.
        
        Args:
            config: MetaDrive configuration dictionary
        """
        super().__init__(config)
    
    def reset(self, seed: Optional[int] = None, **kwargs: Any) -> tuple:
        """Reset the environment with seed modulo handling.
        
        Wraps the seed to valid range if provided to ensure parallel
        environments get different scenarios.
        
        Args:
            seed: Environment seed (will be wrapped to valid range)
            **kwargs: Additional arguments passed to parent reset
            
        Returns:
            Tuple of (observation, info)
        """
        if seed is not None and self.config["num_scenarios"] > 0:
            seed = self.config["start_seed"] + (seed % self.config["num_scenarios"])
        return super().reset(seed=seed, **kwargs)


class TopDownMetaDriveEnvWrapper(TopDownMetaDriveEnvV2):
    """Wrapper for TopDownMetaDriveEnvV2 with seed modulo handling.
    
    This wrapper ensures that when multiple parallel environments are created,
    each gets a different scenario by wrapping the seed to the valid range.
    Used for BEV (bird's eye view) observations.
    
    Attributes:
        config: MetaDrive configuration dictionary
    
    Example:
        >>> config = {"start_seed": 1000, "num_scenarios": 100}
        >>> env = TopDownMetaDriveEnvWrapper(config=config)
        >>> obs, info = env.reset(seed=1500)
    """
    
    def __init__(self, config: dict) -> None:
        """Initialize the wrapper.
        
        Args:
            config: MetaDrive configuration dictionary
        """
        super().__init__(config)
    
    def reset(self, seed: Optional[int] = None, **kwargs: Any) -> tuple:
        """Reset the environment with seed modulo handling.
        
        Wraps the seed to valid range if provided to ensure parallel
        environments get different scenarios.
        
        Args:
            seed: Environment seed (will be wrapped to valid range)
            **kwargs: Additional arguments passed to parent reset
            
        Returns:
            Tuple of (observation, info)
        """
        if seed is not None and self.config["num_scenarios"] > 0:
            seed = self.config["start_seed"] + (seed % self.config["num_scenarios"])
        return super().reset(seed=seed, **kwargs)


class EnvironmentFactory:
    """Factory for creating MetaDrive environments with specific sensor types.
    
    This factory provides a unified interface to create environments for
    different sensor types, handling the specific environment classes and
    configurations required for each.
    
    Supported sensor types:
    - "lidar": Standard MetaDriveEnv with LiDAR
    - "bev": TopDownMetaDriveEnvV2 for BEV observations
    - "rgb": MetaDriveEnv with RGB camera
    - "dense-lidar": MetaDriveEnv with dense LiDAR
    
    Example:
        >>> from src.sac_training.env_factory import EnvironmentFactory
        >>> # Create BEV environment
        >>> env = EnvironmentFactory.create_env("bev")
        >>> obs, info = env.reset()
        
        >>> # Create LiDAR environment with rendering
        >>> env = EnvironmentFactory.create_env("lidar", render=True)
    """
    
    @classmethod
    def create_env(
        cls,
        sensor_type: str,
        render: bool = False,
        **config_overrides: Any,
    ) -> Monitor:
        """Create a MetaDrive environment with the specified sensor type.
        
        Args:
            sensor_type: Type of sensor ("lidar", "bev", "rgb", "dense-lidar")
            render: Whether to enable rendering
            **config_overrides: Optional configuration overrides
            
        Returns:
            Monitored MetaDrive environment
            
        Raises:
            ValueError: If sensor_type is not supported
            
        Example:
            >>> env = EnvironmentFactory.create_env("bev", render=True)
            >>> print(type(env))
            <class 'stable_baselines3.common.monitor.Monitor'>
        """
        # Get sensor-specific configuration
        config = get_sensor_config(sensor_type, **config_overrides)
        
        # Add rendering configuration
        config["use_render"] = render
        if render:
            config["window_size"] = (1200, 800)
        
        # Create appropriate environment based on sensor type
        if sensor_type == "bev":
            env = TopDownMetaDriveEnvWrapper(config=config)
        else:
            env = MetaDriveEnvWrapper(config=config)
        
        # Wrap with Monitor for episode tracking
        env = Monitor(env)
        
        return env
    
    @classmethod
    def get_env_class(cls, sensor_type: str) -> type:
        """Get the environment class for a sensor type.
        
        Args:
            sensor_type: Type of sensor
            
        Returns:
            Environment class (MetaDriveEnvWrapper or TopDownMetaDriveEnvWrapper)
            
        Example:
            >>> env_class = EnvironmentFactory.get_env_class("bev")
            >>> print(env_class.__name__)
            TopDownMetaDriveEnvWrapper
        """
        if sensor_type == "bev":
            return TopDownMetaDriveEnvWrapper
        return MetaDriveEnvWrapper


def create_env(
    sensor_type: str,
    render: bool = False,
    render_mode: str = "rgb_array",
    **config_overrides: Any,
) -> Monitor:
    """Convenience function to create a MetaDrive environment.
    
    This is a shorthand for EnvironmentFactory.create_env().
    
    Args:
        sensor_type: Type of sensor ("lidar", "bev", "rgb", "dense-lidar")
        render: Whether to enable rendering
        render_mode: Rendering mode ("rgb_array" or "onscreen")
        **config_overrides: Optional configuration overrides
        
    Returns:
        Monitored MetaDrive environment
        
    Example:
        >>> from src.sac_training.env_factory import create_env
        >>> env = create_env("lidar", traffic_density=0.2)
    """
    return EnvironmentFactory.create_env(sensor_type, render, render_mode, **config_overrides)
