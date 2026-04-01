"""Sensor factory for creating observation-specific MetaDrive configurations.

This module provides a unified factory interface for creating environment
configurations based on the sensor type. Supports LiDAR, BEV, RGB camera,
and dense LiDAR sensors.

Usage:
    >>> from src.sac_training.sensors import SensorFactory
    >>> config = SensorFactory.get_config("bev")
    >>> print(config["frame_stack"])
    3
"""

from typing import Dict, Any, Optional
from dataclasses import asdict

from src.sac_training.config import (
    LiDARConfig,
    BEVConfig,
    RGBConfig,
    DenseLiDARConfig,
)


class SensorFactory:
    """Factory class for creating sensor-specific MetaDrive configurations.
    
    This factory provides a unified interface to get environment configurations
    for different sensor types. Each sensor type has specific requirements
    for the MetaDrive environment setup.
    
    Supported sensor types:
    - "lidar": Default LiDAR with 240 lasers
    - "bev": Bird's Eye View semantic segmentation (5 channels)
    - "rgb": RGB camera observations
    - "dense-lidar": Dense LiDAR with 720 lasers
    
    Example:
        >>> from src.sac_training.sensors import SensorFactory
        >>> # Get BEV configuration
        >>> bev_config = SensorFactory.get_config("bev")
        >>> print(bev_config["resolution_size"])
        84
        
        >>> # Get LiDAR configuration with overrides
        >>> lidar_config = SensorFactory.get_config("lidar", traffic_density=0.2)
        >>> print(lidar_config["traffic_density"])
        0.2
    """
    
    # Mapping of sensor type names to their configuration classes
    _SENSOR_CONFIGS = {
        "lidar": LiDARConfig,
        "bev": BEVConfig,
        "rgb": RGBConfig,
        "dense-lidar": DenseLiDARConfig,
    }
    
    # Base MetaDrive configuration that applies to all sensor types
    _BASE_CONFIG = {
        "manual_control": False,
        "start_seed": 1000,
        "num_scenarios": 1000,
        "traffic_density": 0.1,
        "horizon": 2000,
        "vehicle_config": {
            "show_lidar": False,
            "show_lane_line_detector": False,
            "show_side_detector": False,
        },
        "driving_reward": 1.0,
        "speed_reward": 0.1,
        "out_of_road_penalty": 5.0,
        "crash_vehicle_penalty": 5.0,
        "crash_object_penalty": 5.0,
        "success_reward": 10.0,
        "out_of_route_done": True,
        "window_size": (800, 600),
    }
    
    @classmethod
    def get_config(
        cls,
        sensor_type: str,
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Get MetaDrive configuration for a specific sensor type.
        
        Args:
            sensor_type: Type of sensor ("lidar", "bev", "rgb", "dense-lidar")
            **overrides: Optional configuration overrides (e.g., traffic_density=0.2)
            
        Returns:
            Dictionary containing MetaDrive configuration
            
        Raises:
            ValueError: If sensor_type is not supported
            
        Example:
            >>> config = SensorFactory.get_config("lidar", traffic_density=0.2)
            >>> print(config["lidar_num_lasers"])
            240
        """
        if sensor_type not in cls._SENSOR_CONFIGS:
            supported = ", ".join(f'"{s}"' for s in cls._SENSOR_CONFIGS.keys())
            raise ValueError(
                f"Unsupported sensor type: '{sensor_type}'. "
                f"Supported types: {supported}"
            )
        
        # Start with base configuration
        config = cls._BASE_CONFIG.copy()
        
        # Get sensor-specific configuration
        sensor_config_class = cls._SENSOR_CONFIGS[sensor_type]
        sensor_config = sensor_config_class()
        
        # Merge sensor-specific settings based on sensor type
        sensor_specific = cls._get_sensor_specific_config(sensor_type, sensor_config)
        
        # For LiDAR configs, merge into vehicle_config["lidar"] instead of top-level
        if sensor_type in ("lidar", "dense-lidar"):
            if "vehicle_config" not in config:
                config["vehicle_config"] = {}
            config["vehicle_config"]["lidar"] = sensor_specific
        # For RGB configs, set up front camera sensor
        elif sensor_type == "rgb":
            from metadrive.component.sensors.rgb_camera import RGBCamera
            
            config["image_observation"] = True
            config["norm_pixel"] = True
            if "vehicle_config" not in config:
                config["vehicle_config"] = {}
            config["vehicle_config"]["image_source"] = "front"
            
            # For RGB, use only the RGB camera sensor (remove default sensors)
            config["sensors"] = {
                "front": (
                    RGBCamera,
                    sensor_config.resolution_size,
                    sensor_config.resolution_size,
                )
            }
        else:
            config.update(sensor_specific)
        
        # Apply overrides
        config.update(overrides)
        
        return config
    
    @classmethod
    def _get_sensor_specific_config(
        cls,
        sensor_type: str,
        sensor_config: Any,
    ) -> Dict[str, Any]:
        """Get sensor-specific configuration settings.
        
        Args:
            sensor_type: Type of sensor
            sensor_config: Sensor configuration instance
            
        Returns:
            Dictionary with sensor-specific settings
        """
        if sensor_type == "lidar":
            return {
                "num_lasers": sensor_config.lidar_num_lasers,
                "distance": sensor_config.lidar_range,
            }
        
        elif sensor_type == "bev":
            return {
                "frame_skip": sensor_config.frame_skip,
                "frame_stack": sensor_config.frame_stack,
                "post_stack": sensor_config.post_stack,
                "norm_pixel": sensor_config.norm_pixel,
                "resolution_size": sensor_config.resolution_size,
                "distance": sensor_config.distance,
            }
        
        elif sensor_type == "rgb":
            return {
                "stack_size": sensor_config.frame_stack,
            }
        
        elif sensor_type == "dense-lidar":
            return {
                "num_lasers": sensor_config.lidar_num_lasers,
                "distance": sensor_config.lidar_range,
            }
        
        return {}
    
    @classmethod
    def get_supported_sensors(cls) -> list[str]:
        """Get list of supported sensor types.
        
        Returns:
            List of supported sensor type strings
            
        Example:
            >>> sensors = SensorFactory.get_supported_sensors()
            >>> print(sensors)
            ['lidar', 'bev', 'rgb', 'dense-lidar']
        """
        return list(cls._SENSOR_CONFIGS.keys())
    
    @classmethod
    def is_supported(cls, sensor_type: str) -> bool:
        """Check if a sensor type is supported.
        
        Args:
            sensor_type: Sensor type string to check
            
        Returns:
            True if sensor type is supported, False otherwise
            
        Example:
            >>> SensorFactory.is_supported("bev")
            True
            >>> SensorFactory.is_supported("unknown")
            False
        """
        return sensor_type in cls._SENSOR_CONFIGS


def get_sensor_config(sensor_type: str, **overrides: Any) -> Dict[str, Any]:
    """Convenience function to get sensor configuration.
    
    This is a shorthand for SensorFactory.get_config().
    
    Args:
        sensor_type: Type of sensor ("lidar", "bev", "rgb", "dense-lidar")
        **overrides: Optional configuration overrides
        
    Returns:
        Dictionary containing MetaDrive configuration
        
    Example:
        >>> from src.sac_training.sensors import get_sensor_config
        >>> config = get_sensor_config("bev")
        >>> print(config["frame_stack"])
        3
    """
    return SensorFactory.get_config(sensor_type, **overrides)


def is_image_observation(sensor_type: str) -> bool:
    """Check if sensor type produces image observations.
    
    Image observations (BEV, RGB) require CnnPolicy and feature extractors,
    while LiDAR observations use MlpPolicy.
    
    Args:
        sensor_type: Type of sensor
        
    Returns:
        True if sensor produces image observations
        
    Example:
        >>> is_image_observation("bev")
        True
        
        >>> is_image_observation("lidar")
        False
    """
    return sensor_type in ("bev", "rgb")


def get_policy_type(sensor_type: str) -> str:
    """Get the appropriate policy type for a sensor.
    
    Image observations (BEV, RGB) use CnnPolicy, while LiDAR uses MlpPolicy.
    
    Args:
        sensor_type: Type of sensor
        
    Returns:
        Policy type string ("CnnPolicy" or "MlpPolicy")
        
    Example:
        >>> get_policy_type("rgb")
        'CnnPolicy'
        
        >>> get_policy_type("lidar")
        'MlpPolicy'
    """
    return "CnnPolicy" if is_image_observation(sensor_type) else "MlpPolicy"
