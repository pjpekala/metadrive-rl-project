"""
Test script to validate MetaDrive observation structure.

This script creates a MetaDrive environment with image observation enabled
and prints detailed information about the observation space, including:
- Dictionary structure (keys, types)
- Image shape, dtype, and value range
- State vector shape, dtype, and value range

Run this BEFORE training to ensure observations are correctly configured.

Usage:
    python test_observation_space.py --modality bev
    python test_observation_space.py --modality rgb
"""

import argparse
import sys
from typing import Any

import numpy as np


def test_observation_space(modality: str = "bev", num_steps: int = 5) -> bool:
    """
    Test the observation space for a given modality.

    Args:
        modality: "bev" or "rgb"
        num_steps: Number of environment steps to test

    Returns:
        True if all tests pass, False otherwise
    """
    print("=" * 70)
    print(f"Testing MetaDrive Observation Space for '{modality}' modality")
    print("=" * 70)

    try:
        from metadrive.envs import MetaDriveEnv
        from metadrive.component.sensors.rgb_camera import RGBCamera
        from metadrive.component.sensors.semantic_camera import SemanticCamera
    except ImportError as e:
        print(f"[ERROR] Failed to import MetaDrive: {e}")
        print("Please install MetaDrive: pip install metadrive-simulator")
        return False

    # Configure environment based on modality
    image_size = 128

    config = {
        "use_render": False,
        "image_observation": True,
        "start_seed": 0,
        "num_scenarios": 1,
        "horizon": 100,
        "traffic_density": 0.0,  # No traffic for quick test
    }

    if modality == "rgb":
        config["sensors"] = {"rgb": (RGBCamera, image_size, image_size)}
        config["vehicle_config"] = {"image_source": "rgb"}
    elif modality == "bev":
        config["sensors"] = {"topdown": (SemanticCamera, image_size, image_size)}
        config["vehicle_config"] = {"image_source": "topdown"}
    else:
        print(f"[ERROR] Unknown modality: {modality}")
        return False

    # Create environment
    print("\n[1] Creating environment...")
    try:
        env = MetaDriveEnv(config=config)
        print(f"    Environment created successfully")
    except Exception as e:
        print(f"[ERROR] Failed to create environment: {e}")
        return False

    # Check observation space
    print("\n[2] Checking observation space...")
    obs_space = env.observation_space
    print(f"    Observation space type: {type(obs_space).__name__}")

    if hasattr(obs_space, "spaces"):
        print(f"    Keys: {list(obs_space.spaces.keys())}")
        for key, space in obs_space.spaces.items():
            print(f"    - {key}:")
            print(f"        Type: {type(space).__name__}")
            print(f"        Shape: {space.shape}")
            print(f"        Dtype: {space.dtype}")
            if hasattr(space, "low") and hasattr(space, "high"):
                print(f"        Low: {np.min(space.low)}, High: {np.max(space.high)}")
    else:
        print(f"    Shape: {obs_space.shape}")
        print(f"    Dtype: {obs_space.dtype}")

    # Reset and check actual observation
    print("\n[3] Resetting environment and checking observation...")
    try:
        result = env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
    except Exception as e:
        print(f"[ERROR] Failed to reset environment: {e}")
        env.close()
        return False

    print(f"    Observation type: {type(obs).__name__}")

    if isinstance(obs, dict):
        print(f"    Keys: {list(obs.keys())}")
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                print(f"    - {key}:")
                print(f"        Shape: {value.shape}")
                print(f"        Dtype: {value.dtype}")
                print(f"        Min: {np.min(value):.4f}, Max: {np.max(value):.4f}")
                print(f"        Mean: {np.mean(value):.4f}, Std: {np.std(value):.4f}")
            else:
                print(f"    - {key}: {type(value).__name__}")
    else:
        print(f"    Shape: {obs.shape}")
        print(f"    Dtype: {obs.dtype}")
        print(f"    Min: {np.min(obs):.4f}, Max: {np.max(obs):.4f}")

    # Test a few steps
    print(f"\n[4] Testing {num_steps} environment steps...")
    for step in range(num_steps):
        action = env.action_space.sample()
        try:
            result = env.step(action)
            if len(result) == 4:
                obs, reward, done, info = result
            else:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated

            if isinstance(obs, dict) and "image" in obs:
                img = obs["image"]
                state = obs.get("state", None)
                state_info = f", state: {state.shape}" if state is not None else ""
                print(
                    f"    Step {step + 1}: image: {img.shape}, "
                    f"range: [{np.min(img):.2f}, {np.max(img):.2f}]{state_info}, "
                    f"reward: {reward:.2f}, done: {done}"
                )
            else:
                print(
                    f"    Step {step + 1}: obs: {obs.shape}, reward: {reward:.2f}, done: {done}"
                )

            if done:
                result = env.reset()
                if isinstance(result, tuple):
                    obs, info = result
                else:
                    obs = result
                print(f"    Episode done, reset environment")

        except Exception as e:
            print(f"[ERROR] Step {step + 1} failed: {e}")
            env.close()
            return False

    # Check action space
    print("\n[5] Checking action space...")
    action_space = env.action_space
    print(f"    Action space type: {type(action_space).__name__}")
    print(f"    Shape: {action_space.shape}")
    if hasattr(action_space, "low") and hasattr(action_space, "high"):
        print(f"    Low: {action_space.low}")
        print(f"    High: {action_space.high}")

    # Close environment
    env.close()

    # Validation checks
    print("\n[6] Validation...")
    all_passed = True

    # Check if observation is dict with expected keys
    if isinstance(obs, dict):
        if "image" in obs:
            print("    [PASS] Observation contains 'image' key")
        else:
            print("    [FAIL] Observation missing 'image' key")
            all_passed = False

        if "state" in obs:
            print("    [PASS] Observation contains 'state' key")
            state_dim = obs["state"].shape[0]
            print(f"           State vector dimension: {state_dim}")
        else:
            print("    [WARN] Observation missing 'state' key (might need to enable)")

        # Check image shape
        if "image" in obs:
            img_shape = obs["image"].shape
            if len(img_shape) == 3:
                h, w, c = img_shape
                print(f"    [INFO] Image shape: (H={h}, W={w}, C={c}) - HWC format")
            elif len(img_shape) == 4:
                h, w, c, stack = img_shape
                print(
                    f"    [INFO] Image shape: (H={h}, W={w}, C={c}, stack={stack}) - HWC+stack format"
                )
            else:
                print(f"    [WARN] Unexpected image shape: {img_shape}")
    else:
        print("    [WARN] Observation is not a dictionary")

    print("\n" + "=" * 70)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 70)

    return all_passed


def test_state_vector_details(modality: str = "bev") -> None:
    """
    Detailed analysis of the state vector contents.

    This function attempts to understand what each element of the
    state vector represents by observing changes during driving.
    """
    print("\n" + "=" * 70)
    print(f"Analyzing State Vector for '{modality}' modality")
    print("=" * 70)

    try:
        from metadrive.envs import MetaDriveEnv
        from metadrive.component.sensors.rgb_camera import RGBCamera
        from metadrive.component.sensors.semantic_camera import SemanticCamera
    except ImportError:
        print("[ERROR] MetaDrive not available")
        return

    image_size = 128
    config = {
        "use_render": False,
        "image_observation": True,
        "start_seed": 0,
        "num_scenarios": 1,
        "horizon": 200,
        "traffic_density": 0.0,
    }

    if modality == "rgb":
        config["sensors"] = {"rgb": (RGBCamera, image_size, image_size)}
        config["vehicle_config"] = {"image_source": "rgb"}
    else:
        config["sensors"] = {"topdown": (SemanticCamera, image_size, image_size)}
        config["vehicle_config"] = {"image_source": "topdown"}

    env = MetaDriveEnv(config=config)

    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result

    if not isinstance(obs, dict) or "state" not in obs:
        print("[ERROR] No state vector in observation")
        env.close()
        return

    state_dim = obs["state"].shape[0]
    print(f"\nState vector dimension: {state_dim}")

    # Collect state vectors over time
    print("\nCollecting state vectors over 50 steps...")
    states = [obs["state"].copy()]

    for i in range(50):
        # Apply constant forward acceleration
        action = np.array([0.5, 0.0])  # throttle, steering
        result = env.step(action)
        obs = result[0] if len(result) == 4 else result[0]

        if isinstance(obs, dict) and "state" in obs:
            states.append(obs["state"].copy())

    states = np.array(states)

    print(f"\nState vector statistics over {len(states)} timesteps:")
    print("-" * 70)
    print(f"{'Idx':<4} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Change':>10}")
    print("-" * 70)

    for i in range(state_dim):
        col = states[:, i]
        mean = np.mean(col)
        std = np.std(col)
        min_val = np.min(col)
        max_val = np.max(col)
        change = np.abs(col[-1] - col[0])

        print(
            f"{i:<4} {mean:>10.4f} {std:>10.4f} {min_val:>10.4f} {max_val:>10.4f} {change:>10.4f}"
        )

    print("-" * 70)
    print("\nInterpretation hints:")
    print("  - Large 'Change' values indicate elements that vary during driving")
    print("  - Small 'Std' values suggest constants or slowly changing values")
    print("  - Elements with high variance likely encode velocity, position, etc.")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MetaDrive observation space")
    parser.add_argument(
        "--modality",
        type=str,
        default="bev",
        choices=["bev", "rgb"],
        help="Sensor modality to test",
    )
    parser.add_argument(
        "--analyze-state",
        action="store_true",
        help="Run detailed state vector analysis",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of environment steps to test",
    )

    args = parser.parse_args()

    success = test_observation_space(modality=args.modality, num_steps=args.steps)

    if args.analyze_state:
        test_state_vector_details(modality=args.modality)

    sys.exit(0 if success else 1)
