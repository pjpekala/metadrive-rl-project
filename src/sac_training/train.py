"""MetaDrive RL Training with SAC Algorithm - Enhanced with Live Rendering & Video Recording.

==========================================================================================
Features:
- Live rendering during training (watch the car drive in real-time!)
- Automatic video recording of best episodes
- Real-time training visualization
- TensorBoard monitoring
- Multi-sensor support (LiDAR, BEV, RGB camera, Dense LiDAR)
- Multi-GPU support

Usage Examples:

# Train with default LiDAR observations (fast, recommended)
uv run src/sac_training/train.py --sensor-type lidar --n-envs 8

# Train with BEV image observations
uv run src/sac_training/train.py --sensor-type bev --n-envs 8

# Train with RGB camera observations
uv run src/sac_training/train.py --sensor-type rgb --n-envs 8

# Train with dense LiDAR observations
uv run src/sac_training/train.py --sensor-type dense-lidar --n-envs 8

# Watch training live (single env, slower but you see everything)
uv run src/sac_training/train.py --sensor-type lidar --live-render --n-envs 1

# Fast training with periodic video recording
uv run src/sac_training/train.py --sensor-type lidar --record-video --video-interval 50 --n-envs 4

# Both live plots and training monitoring
uv run src/sac_training/train.py --sensor-type lidar --visualize --live-render --n-envs 1
"""

import os
import argparse
import time
from pathlib import Path
from typing import Optional, Any

import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from src.sac_training.config import TrainingConfig
from src.sac_training.sensors import get_sensor_config, get_policy_type, is_image_observation
from src.sac_training.env_factory import EnvironmentFactory
from src.sac_training.feature_extractors import get_feature_extractor_config
from src.sac_training.callbacks import VideoRecorderCallback, VisualizationCallback


class SACTrainer:
    """SAC Trainer for MetaDrive with modular sensor support.
    
    This class encapsulates the SAC training logic and provides a clean
    interface for training agents with different sensor types.
    
    Attributes:
        sensor_type: Type of sensor ("lidar", "bev", "rgb", "dense-lidar")
        config: Training configuration
        save_dir: Directory to save models
        device: Device to use for training
    
    Example:
        >>> from src.sac_training.train import SACTrainer
        >>> trainer = SACTrainer(sensor_type="bev")
        >>> model = trainer.train()
    """
    
    def __init__(
        self,
        sensor_type: str = "lidar",
        training_config: Optional[TrainingConfig] = None,
        save_dir: Optional[str] = None,
        device: str = "auto",
    ) -> None:
        """Initialize the SAC trainer.
        
        Args:
            sensor_type: Type of sensor ("lidar", "bev", "rgb", "dense-lidar")
            training_config: Training configuration (uses defaults if None)
            save_dir: Directory to save models (auto-generated if None)
            device: Device to use ("auto", "cuda", "cpu", "mps")
        """
        self.sensor_type = sensor_type
        self.config = training_config or TrainingConfig()
        self.device = self._get_device(device)
        
        # Auto-generate save directory
        if save_dir is None:
            save_dir = f"models/sac_metadrive_{sensor_type}"
        self.save_dir = save_dir
        
        # Create directories
        self.log_dir = os.path.join(self.save_dir, "logs")
        self.checkpoint_dir = os.path.join(self.save_dir, "checkpoints")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def _get_device(self, device: str) -> str:
        """Determine the device to use for training.
        
        Args:
            device: Device preference ("auto", "cuda", "cpu", "mps")
            
        Returns:
            Device string
        """
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            #     return "mps"
            else:
                return "cpu"
        return device
    
    def create_training_env(
        self,
        render: bool = False,
        n_envs: int = 1,
    ) -> Any:
        """Create training environment(s).
        
        Args:
            render: Whether to enable rendering
            n_envs: Number of parallel environments
            
        Returns:
            Vectorized environment
        """
        if render:
            # For live rendering, use single env with onscreen rendering
            env = DummyVecEnv([
                lambda: EnvironmentFactory.create_env(self.sensor_type, render=True)
            ])
        elif n_envs > 1:
            # For parallel training, use SubprocVecEnv
            env = make_vec_env(
                lambda: EnvironmentFactory.create_env(self.sensor_type),
                n_envs=n_envs,
                seed=0,
                vec_env_cls=SubprocVecEnv,
            )
        else:
            env = DummyVecEnv([
                lambda: EnvironmentFactory.create_env(self.sensor_type)
            ])
        
        return env
    
    def train(
        self,
        visualize: bool = False,
        live_render: bool = False,
        record_video: bool = False,
        video_interval: int = 50,
        plot_interval: int = 100,
    ) -> SAC:
        """Train the SAC agent.
        
        Args:
            visualize: Enable live training plots
            live_render: Show live rendering window
            record_video: Record training videos
            video_interval: Record video every N episodes
            plot_interval: Update plots every N steps
            
        Returns:
            Trained SAC model
        """
        print("=" * 70)
        print("MetaDrive SAC Training")
        print("=" * 70)
        print(f"Sensor Type: {self.sensor_type}")
        print(f"Device: {self.device}")
        print(f"Observation: {self.sensor_type.upper()}")
        print(f"Number of parallel environments: {self.config.n_envs}")
        print(f"Live rendering: {'YES' if live_render else 'No'}")
        print(f"Video recording: {'YES' if record_video else 'No'}")
        print(f"Live plots: {'Enabled' if visualize else 'Disabled'}")
        print("=" * 70)
        
        # Create training environment
        env = self.create_training_env(
            render=live_render,
            n_envs=1 if live_render else self.config.n_envs,
        )
        
        # Create separate environment for video recording
        video_env = None
        if record_video:
            video_dir = os.path.join(self.save_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            video_env = EnvironmentFactory.create_env(self.sensor_type, render=True)
        
        print(f"\nObservation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Get policy and feature extractor configuration
        policy = get_policy_type(self.sensor_type)
        policy_kwargs = get_feature_extractor_config(self.sensor_type, features_dim=256)
        
        # Adjust settings for image observations (memory optimization)
        effective_buffer_size = self.config.buffer_size
        effective_batch_size = self.config.batch_size
        
        if is_image_observation(self.sensor_type):
            effective_buffer_size = min(self.config.buffer_size, 50_000)
            effective_batch_size = min(self.config.batch_size, 128)
            print(f"Image observation: buffer_size={effective_buffer_size:,}, batch_size={effective_batch_size}")
        
        # Initialize SAC agent
        model = SAC(
            policy,
            env,
            learning_rate=self.config.learning_rate,
            buffer_size=effective_buffer_size,
            learning_starts=self.config.learning_starts,
            batch_size=effective_batch_size,
            tau=self.config.tau,
            gamma=self.config.gamma,
            train_freq=self.config.train_freq,
            gradient_steps=self.config.gradient_steps,
            ent_coef=self.config.ent_coef,
            verbose=self.config.verbose,
            tensorboard_log=self.log_dir,
            device=self.device,
            policy_kwargs=policy_kwargs,
        )
        
        print(f"\nUsing device: {model.device}")
        print(f"Policy network: {type(model.policy).__name__}")
        
        # Setup callbacks
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.save_freq // self.config.n_envs,
            save_path=self.checkpoint_dir,
            name_prefix="sac_metadrive",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )
        callbacks.append(checkpoint_callback)
        
        # Visualization callback
        if visualize:
            viz_callback = VisualizationCallback(
                render_env=None,
                render_interval=999999,
                plot_interval=plot_interval,
                verbose=1,
            )
            callbacks.append(viz_callback)
        
        # Video recording callback
        if record_video and video_env is not None:
            video_callback = VideoRecorderCallback(
                eval_env=video_env,
                video_folder=os.path.join(self.save_dir, "videos"),
                record_interval=video_interval,
                verbose=1,
            )
            callbacks.append(video_callback)
        
        print("\n" + "=" * 70)
        print("Starting Training...")
        print("=" * 70)
        print(f"Total timesteps: {self.config.total_timesteps:,}")
        print(f"Checkpoint every: {self.config.save_freq:,} steps")
        if record_video:
            print(f"Recording video every: {video_interval} episodes")
        if visualize:
            print(f"Plot update every: {plot_interval} steps")
        print(f"Logs directory: {self.log_dir}")
        print("=" * 70 + "\n")
        
        # Train the agent
        try:
            model.learn(
                total_timesteps=self.config.total_timesteps,
                callback=callbacks,
                log_interval=4,
                progress_bar=True,
            )
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user!")
        finally:
            # Save final model
            final_model_path = os.path.join(self.save_dir, "sac_metadrive_final")
            model.save(final_model_path)
            print(f"\nFinal model saved to: {final_model_path}")
            
            # Close environments
            env.close()
            if video_env is not None:
                video_env.close()
        
        print("\n" + "=" * 70)
        print("Training Complete!")
        if record_video:
            print(f"📹 Videos saved to: {os.path.join(self.save_dir, 'videos')}")
        print("=" * 70)
        
        return model


def test_trained_agent(
    model_path: str,
    sensor_type: str = "lidar",
    num_episodes: int = 10,
    render: bool = True,
) -> None:
    """Test a trained agent with live rendering.
    
    Args:
        model_path: Path to the saved model
        sensor_type: Type of sensor used during training
        num_episodes: Number of test episodes
        render: Whether to render the environment
    """
    print("\n" + "=" * 70)
    print("Testing Trained Agent...")
    print("=" * 70)
    
    # Create test environment
    env = EnvironmentFactory.create_env(sensor_type, render=render)
    
    # Load trained model
    model = SAC.load(model_path)
    print(f"Model loaded from: {model_path}")
    print(f"Device: {model.device}")
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        print(f"\n[Episode {episode + 1}/{num_episodes}] Starting...")
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += float(reward)
            episode_length += 1
            
            if render:
                time.sleep(0.01)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if info.get("arrive_dest", False):
            success_count += 1
            print(f"✓ SUCCESS! Reward: {episode_reward:.2f}, Length: {episode_length}")
        else:
            crash = info.get("crash", False)
            out_of_road = info.get("out_of_road", False)
            reason = "Crash" if crash else ("Out of road" if out_of_road else "Max steps")
            print(f"✗ Failed ({reason}). Reward: {episode_reward:.2f}, Length: {episode_length}")
    
    env.close()
    
    # Print statistics
    print("\n" + "=" * 70)
    print("Test Results Summary:")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Success rate: {success_count / num_episodes * 100:.1f}%")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SAC agent on MetaDrive with multiple sensor types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default LiDAR (fastest)
  uv run src/sac_training/train.py --sensor-type lidar --n-envs 8
  
  # Train with BEV image observations
  uv run src/sac_training/train.py --sensor-type bev --n-envs 8
  
  # Train with RGB camera observations
  uv run src/sac_training/train.py --sensor-type rgb --n-envs 8
  
  # Train with dense LiDAR observations
  uv run src/sac_training/train.py --sensor-type dense-lidar --n-envs 8
  
  # Watch training live (best for learning, slower)
  uv run src/sac_training/train.py --sensor-type lidar --live-render --n-envs 1
  
  # Fast training with video recording every 50 episodes
  uv run src/sac_training/train.py --sensor-type lidar --record-video --video-interval 50 --n-envs 4
  
  # Test a trained model
  uv run -m src.sac_training.train --test models/sac_metadrive_lidar/sac_metadrive_final --sensor-type lidar
  
  # Legacy: use --bev flag (maps to --sensor-type bev)
  uv run src/sac_training/train.py --bev --n-envs 8
        """,
    )
    
    # Sensor type
    parser.add_argument(
        "--sensor-type",
        type=str,
        default="lidar",
        choices=["lidar", "bev", "rgb", "dense-lidar"],
        help="Sensor type for observations (default: lidar)",
    )
    
    # Legacy backward compatibility
    parser.add_argument(
        "--bev",
        action="store_true",
        help="Use BEV observations (equivalent to --sensor-type bev)",
    )
    
    # Training parameters
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1_000_000,
        help="Replay buffer size",
    )
    
    # Visualization options
    parser.add_argument(
        "--live-render",
        action="store_true",
        help="Show live rendering during training",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record videos of training episodes",
    )
    parser.add_argument(
        "--video-interval",
        type=int,
        default=50,
        help="Record video every N episodes",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable live training plots",
    )
    parser.add_argument(
        "--plot-interval",
        type=int,
        default=200,
        help="Update training plots every N steps",
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto (default), cuda, cpu, or mps",
    )
    
    # Testing
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Test a trained model (provide path)",
    )
    parser.add_argument(
        "--test-episodes",
        type=int,
        default=10,
        help="Number of test episodes",
    )
    
    args = parser.parse_args()
    
    # Handle backward compatibility: --bev flag
    if args.bev:
        args.sensor_type = "bev"
    
    if args.test:
        # Test mode
        test_trained_agent(
            model_path=args.test,
            sensor_type=args.sensor_type,
            num_episodes=args.test_episodes,
            render=True,
        )
    else:
        # Training mode
        training_config = TrainingConfig(
            total_timesteps=args.total_timesteps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            n_envs=args.n_envs,
        )
        
        trainer = SACTrainer(
            sensor_type=args.sensor_type,
            training_config=training_config,
            device=args.device,
        )
        
        trained_model = trainer.train(
            visualize=args.visualize,
            live_render=args.live_render,
            record_video=args.record_video,
            video_interval=args.video_interval,
            plot_interval=args.plot_interval,
        )
        
        print("\n\n" + "=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print(f"To test the trained agent with live rendering:")
        print(f"  uv run -m src.sac_training.train --test {trainer.save_dir}/sac_metadrive_final --sensor-type {args.sensor_type}")
        if args.record_video:
            print(f"\nVideos saved in: {trainer.save_dir}/videos/")
        print("=" * 70)
