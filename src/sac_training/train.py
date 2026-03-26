"""
MetaDrive RL Training with SAC Algorithm - Enhanced with Live Rendering & Video Recording
==========================================================================================
Features:
- Live rendering during training (watch the car drive in real-time!)
- Automatic video recording of best episodes
- Real-time training visualization
- TensorBoard monitoring
- Multi-GPU support

Installation:
pip install metadrive-simulator
pip install stable-baselines3[extra]
pip install tensorboard
pip install matplotlib
pip install imageio
pip install imageio-ffmpeg

Usage Examples:
# Watch training live (single env, slower but you see everything)
uv run src/sac_training/train.py --live-render --n-envs 1

# Fast training with periodic video recording
uv run src/sac_training/train.py --record-video --video-interval 50 --n-envs 4

# Both live plots and training monitoring
uv run src/sac_training/train.py --visualize --live-render --n-envs 1
"""

import os
import argparse
import time
from pathlib import Path
from dataclasses import dataclass
from metadrive.envs import MetaDriveEnv
from metadrive.envs.top_down_env import TopDownMetaDriveEnvV2
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import numpy as np
import matplotlib.pyplot as plt
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    print("Warning: imageio not available. Video recording disabled.")


@dataclass
class BEVSensorConfig:
    """Configuration for BEV (bird's-eye view) sensor."""

    frame_skip: int = 5
    frame_stack: int = 3
    post_stack: int = 5
    norm_pixel: bool = True
    resolution_size: int = 84
    distance: int = 30


class VideoRecorderCallback(BaseCallback):
    """
    Callback for recording videos of the agent during training
    """
    def __init__(self, eval_env, video_folder, record_interval=50, video_length=500, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.video_folder = Path(video_folder)
        self.video_folder.mkdir(parents=True, exist_ok=True)
        self.record_interval = record_interval  # Record every N episodes
        self.video_length = video_length  # Max steps per video
        self.episode_count = 0
        self.best_reward = -float('inf')
        
    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals['dones'][0]:
            self.episode_count += 1
            
            # Record video at intervals
            if self.episode_count % self.record_interval == 0:
                reward = self._record_video(deterministic=True)
                
                # Save best episodes
                if reward > self.best_reward:
                    self.best_reward = reward
                    if self.verbose > 0:
                        print(f"\n🎬 New best episode! Reward: {reward:.2f}")
                    self._record_video(deterministic=True, prefix="best")
                    
        return True
    
    def _record_video(self, deterministic=True, prefix="episode"):
        """Record a single episode as video"""
        if not IMAGEIO_AVAILABLE:
            return 0
            
        frames = []
        obs = self.eval_env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < self.video_length:
            # Render frame
            frame = self.eval_env.render(mode='rgb_array')
            if frame is not None:
                frames.append(frame)
            
            # Take action
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        # Save video
        if frames:
            video_path = self.video_folder / f"{prefix}_ep{self.episode_count}_r{total_reward:.0f}.mp4"
            imageio.mimsave(str(video_path), frames, fps=30)
            if self.verbose > 0:
                print(f"📹 Video saved: {video_path} (Reward: {total_reward:.2f})")
        
        return total_reward


class VisualizationCallback(BaseCallback):
    """
    Callback for visualizing training progress in real-time
    """
    def __init__(self, render_env=None, render_interval=10, plot_interval=100, verbose=0):
        super().__init__(verbose)
        self.render_env = render_env
        self.render_interval = render_interval
        self.plot_interval = plot_interval
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.episode_count = 0
        
        # For live plotting
        self.fig = None
        self.axes = None
        
    def _on_training_start(self):
        if self.verbose > 0:
            print("\n[Visualization] Starting training visualization...")
            
    def _on_step(self) -> bool:
        # Track episode statistics
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # Check if episode ended
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Check if agent succeeded
            info = self.locals.get('infos', [{}])[0]
            success = info.get('arrive_dest', False)
            self.episode_successes.append(1.0 if success else 0.0)
            
            self.episode_count += 1
            
            if self.verbose > 0:
                success_rate = np.mean(self.episode_successes[-100:]) * 100 if self.episode_successes else 0
                avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                print(f"\n[Episode {self.episode_count}] "
                      f"Reward: {self.current_episode_reward:.2f} | "
                      f"Length: {self.current_episode_length} | "
                      f"Success: {'✓' if success else '✗'} | "
                      f"Avg Reward (100): {avg_reward:.2f} | "
                      f"Success Rate (100): {success_rate:.1f}%")
            
            # Reset episode tracking
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
        # Update plots periodically
        if self.num_timesteps % self.plot_interval == 0 and len(self.episode_rewards) > 0:
            self._update_plots()
            
        return True
    
    def _update_plots(self):
        """Update live training plots"""
        if len(self.episode_rewards) < 2:
            return
            
        if self.fig is None:
            plt.ion()
            self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
            self.fig.suptitle('MetaDrive Training Progress', fontsize=16)
        
        # Clear all subplots
        for ax in self.axes.flat:
            ax.clear()
        
        episodes = np.arange(len(self.episode_rewards))
        
        # Plot 1: Episode Rewards
        ax1 = self.axes[0, 0]
        ax1.plot(episodes, self.episode_rewards, alpha=0.3, label='Raw')
        if len(self.episode_rewards) >= 10:
            smooth_rewards = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
            ax1.plot(episodes[9:], smooth_rewards, linewidth=2, label='Smoothed (10)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Episode Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Episode Lengths
        ax2 = self.axes[0, 1]
        ax2.plot(episodes, self.episode_lengths, alpha=0.3, label='Raw')
        if len(self.episode_lengths) >= 10:
            smooth_lengths = np.convolve(self.episode_lengths, np.ones(10)/10, mode='valid')
            ax2.plot(episodes[9:], smooth_lengths, linewidth=2, label='Smoothed (10)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Episode Lengths')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Success Rate
        ax3 = self.axes[1, 0]
        if len(self.episode_successes) >= 10:
            window = min(100, len(self.episode_successes))
            success_rates = [np.mean(self.episode_successes[max(0, i-window):i+1]) * 100 
                           for i in range(len(self.episode_successes))]
            ax3.plot(episodes, success_rates, linewidth=2)
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Success Rate (%)')
            ax3.set_title(f'Success Rate (rolling window={window})')
            ax3.set_ylim([0, 105])
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Statistics Summary
        ax4 = self.axes[1, 1]
        ax4.axis('off')
        recent = min(100, len(self.episode_rewards))
        stats_text = f"""
        Training Statistics (Last {recent} episodes)
        {'='*45}
        Episodes Completed: {len(self.episode_rewards)}
        Total Steps: {self.num_timesteps:,}
        
        Average Reward: {np.mean(self.episode_rewards[-recent:]):.2f}
        Best Reward: {np.max(self.episode_rewards):.2f}
        
        Average Length: {np.mean(self.episode_lengths[-recent:]):.1f}
        
        Success Rate: {np.mean(self.episode_successes[-recent:]) * 100:.1f}%
        """
        ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        plt.pause(0.001)


class MetaDriveEnvWrapper(MetaDriveEnv):
    """Wrapper to handle seed modulo operation for parallel environments"""
    def reset(self, seed=None, **kwargs):
        # Wrap seed to valid range if provided
        if seed is not None and self.config["num_scenarios"] > 0:
            seed = self.config["start_seed"] + (seed % self.config["num_scenarios"])
        return super().reset(seed=seed, **kwargs)


def create_metadrive_env(render=False, render_mode='rgb_array'):
    """
    Create a MetaDrive environment with optional rendering
    
    Args:
        render: Whether to render the environment
        render_mode: 'rgb_array' for video recording, 'onscreen' for live window
    """
    config = dict(
        # Environment settings
        use_render=render,
        manual_control=False,
        
        # Map configuration - enough scenarios for parallel environments
        start_seed=1000,
        num_scenarios=1000,
        traffic_density=0.1,
        
        # Episode settings
        horizon=2000,
        
        # Vehicle configuration
        vehicle_config=dict(
            show_lidar=False,
            show_lane_line_detector=False,
            show_side_detector=False,
        ),
        
        # Observation space
        image_observation=False,
        
        # Reward shaping
        driving_reward=1.0,
        speed_reward=0.1,
        out_of_road_penalty=5.0,
        crash_vehicle_penalty=5.0,
        crash_object_penalty=5.0,
        success_reward=10.0,
        out_of_route_done=True,
        
        # Rendering settings (when render=True)
        window_size=(1200, 800) if render else (800, 600),
        camera_height=3.0,
        camera_dist=8.0,
    )
    
    env = MetaDriveEnvWrapper(config=config)
    env = Monitor(env)
    return env


def create_bev_env(
    render: bool = False,
    render_mode: str = 'rgb_array',
    bev_config: BEVSensorConfig = None,
):
    """
    Create a MetaDrive environment with BEV (bird's-eye view) image observations.
    
    Args:
        render: Whether to render the environment
        render_mode: 'rgb_array' for video recording, 'onscreen' for live window
        bev_config: BEV sensor configuration (uses defaults if None)
    """
    if bev_config is None:
        bev_config = BEVSensorConfig()
    
    config = dict(
        # BEV sensor configuration
        frame_skip=bev_config.frame_skip,
        frame_stack=bev_config.frame_stack,
        post_stack=bev_config.post_stack,
        norm_pixel=bev_config.norm_pixel,
        resolution_size=bev_config.resolution_size,
        distance=bev_config.distance,
        
        # Environment settings
        use_render=render,
        manual_control=False,
        
        # Map configuration
        start_seed=1000,
        num_scenarios=1000,
        traffic_density=0.1,
        
        # Episode settings
        horizon=2000,
        
        # Reward shaping (same as original SAC)
        driving_reward=1.0,
        speed_reward=0.1,
        out_of_road_penalty=5.0,
        crash_vehicle_penalty=5.0,
        crash_object_penalty=5.0,
        success_reward=10.0,
        out_of_route_done=True,
        
        # Rendering settings (when render=True)
        window_size=(1200, 800) if render else (800, 600),
    )
    
    env = TopDownMetaDriveEnvV2(config=config)
    env = Monitor(env)
    return env


def train_sac_agent(
    total_timesteps=1000_000,
    learning_rate=3e-4,
    buffer_size=1000_000,
    learning_starts=1000,
    batch_size=256,
    n_envs=8,
    save_dir="models/sac_metadrive",
    save_freq=120_000,
    visualize=False,
    live_render=False,
    record_video=False,
    video_interval=50,
    plot_interval=100,
    device='cuda',
    use_bev=False,
    bev_config=None,
):
    """
    Train a SAC agent on MetaDrive with visualization options
    
    Args:
        use_bev: Use BEV (bird's-eye view) image observations instead of default LiDAR
        bev_config: BEV sensor configuration (uses defaults if None and use_bev=True)
    """
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(save_dir, "logs")
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    video_dir = os.path.join(save_dir, "videos")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    if record_video:
        os.makedirs(video_dir, exist_ok=True)
    
    print("=" * 70)
    print("MetaDrive SAC Training")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Observation: {'BEV (84x84 image)' if use_bev else 'LiDAR (default)'}")
    print(f"Number of parallel environments: {n_envs}")
    print(f"Live rendering: {'YES - Training window will show car driving!' if live_render else 'No'}")
    print(f"Video recording: {'YES - Best episodes saved to ' + video_dir if record_video else 'No'}")
    print(f"Live plots: {'Enabled' if visualize else 'Disabled'}")
    if live_render and n_envs > 1:
        print("\n⚠️  WARNING: Live rendering works best with --n-envs 1")
        print("   Multiple envs will only show the first environment")
    print("=" * 70)
    
    # Determine which environment factory to use
    if use_bev:
        env_factory = lambda render=False, mode='rgb_array': create_bev_env(
            render=render, render_mode=mode, bev_config=bev_config
        )
    else:
        env_factory = create_metadrive_env
    
    # Create training environments
    print("\nCreating training environments...")
    if live_render:
        # For live rendering, use single env with onscreen rendering
        print("🎮 Setting up LIVE RENDERING - You'll see the car drive!")
        env = DummyVecEnv([lambda: env_factory(render=True, render_mode='onscreen')])
    elif n_envs > 1:
        # For parallel training, MUST use SubprocVecEnv (MetaDrive doesn't support multiple envs in same process)
        print(f"Using SubprocVecEnv with {n_envs} parallel environments")
        env = make_vec_env(
            lambda: env_factory(render=False),
            n_envs=n_envs,
            seed=0,
            vec_env_cls=SubprocVecEnv
        )
    else:
        env = DummyVecEnv([lambda: env_factory(render=False)])
    
    # Create separate environment for video recording
    video_env = None
    if record_video and IMAGEIO_AVAILABLE:
        print("📹 Setting up video recording environment...")
        video_env = env_factory(render=True, render_mode='rgb_array')
    
    print("\nEnvironment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    print("\n" + "=" * 70)
    print("Initializing SAC Agent...")
    print("=" * 70)
    
    # Choose policy based on observation type
    policy = "CnnPolicy" if use_bev else "MlpPolicy"
    print(f"Using policy: {policy}")
    
    # Initialize SAC agent
    model = SAC(
        policy,
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        verbose=1,
        tensorboard_log=log_dir,
        device=device,
    )
    
    print(f"\nUsing device: {model.device}")
    print(f"Policy network: {type(model.policy).__name__}")
    
    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path=checkpoint_dir,
        name_prefix="sac_metadrive",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Visualization callback
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
            video_folder=video_dir,
            record_interval=video_interval,
            verbose=1,
        )
        callbacks.append(video_callback)
    
    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Checkpoint every: {save_freq:,} steps")
    if record_video:
        print(f"Recording video every: {video_interval} episodes")
    if visualize:
        print(f"Plot update every: {plot_interval} steps")
    print(f"Logs directory: {log_dir}")
    print("\nTo monitor training with TensorBoard, run:")
    print(f"  tensorboard --logdir {log_dir}")
    if live_render:
        print("\n🎮 LIVE RENDERING ENABLED - Watch the car learn to drive!")
    print("=" * 70 + "\n")
    
    # Train the agent
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=4,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    finally:
        if viz_callback.fig is not None:
            plt.ioff()
            plt.savefig(os.path.join(save_dir, 'training_progress.png'), dpi=150)
            print(f"\nTraining plots saved to: {os.path.join(save_dir, 'training_progress.png')}")
    
    # Save final model
    final_model_path = os.path.join(save_dir, "sac_metadrive_final")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Close environments
    env.close()
    if video_env is not None:
        video_env.close()
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    if record_video:
        print(f"📹 Videos saved to: {video_dir}")
    print("=" * 70)
    
    return model


def test_trained_agent(model_path, num_episodes=10, render=True):
    """
    Test a trained agent with live rendering
    """
    print("\n" + "=" * 70)
    print("Testing Trained Agent...")
    print("=" * 70)
    
    # Create test environment with rendering
    env = create_metadrive_env(render=render, render_mode='onscreen')
    
    # Load trained model
    model = SAC.load(model_path)
    print(f"Model loaded from: {model_path}")
    print(f"Device: {model.device}")
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        print(f"\n[Episode {episode + 1}/{num_episodes}] Starting...")
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                time.sleep(0.01)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if info.get('arrive_dest', False):
            success_count += 1
            print(f"✓ SUCCESS! Reward: {episode_reward:.2f}, Length: {episode_length}")
        else:
            crash = info.get('crash', False)
            out_of_road = info.get('out_of_road', False)
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
        description='Train SAC agent on MetaDrive with live rendering and video recording',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Watch training live (best for learning, slower)
  uv run src/sac_training/train.py --live-render --n-envs 1 --total-timesteps 50000
  
  # Fast training with video recording every 50 episodes
  uv run src/sac_training/train.py --record-video --video-interval 50 --n-envs 4
  
  # BEV image observations training
  uv run src/sac_training/train.py --bev --n-envs 4
  
  # Everything enabled (live render + videos + plots)
  uv run src/sac_training/train.py --live-render --record-video --visualize --n-envs 1
  
  # Test a trained model
  uv run src/sac_training/train.py --test models/sac_metadrive/sac_metadrive_final
        """)
    
    parser.add_argument('--total-timesteps', type=int, default=1000_000,
                       help='Total training timesteps')
    parser.add_argument('--n-envs', type=int, default=8,
                       help='Number of parallel environments')
    parser.add_argument('--live-render', action='store_true',
                       help='🎮 Show live rendering during training (see the car drive!)')
    parser.add_argument('--record-video', action='store_true',
                       help='📹 Record videos of training episodes')
    parser.add_argument('--video-interval', type=int, default=50,
                       help='Record video every N episodes')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable live training plots')
    parser.add_argument('--plot-interval', type=int, default=200,
                       help='Update training plots every N steps')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: auto, cuda, cpu, or cuda:0, cuda:1, etc.')
    parser.add_argument('--test', type=str, default=None,
                       help='Test a trained model (provide path)')
    parser.add_argument('--test-episodes', type=int, default=10,
                       help='Number of test episodes')
    parser.add_argument('--bev', action='store_true',
                       help='Use BEV (bird\'s-eye view) image observations instead of default LiDAR')
    
    args = parser.parse_args()
    
    if args.test:
        # Test mode
        test_trained_agent(args.test, num_episodes=args.test_episodes, render=True)
    else:
        # Training mode
        if args.live_render:
            print("\n" + "="*70)
            print("🎮 LIVE RENDERING MODE")
            print("="*70)
            print("You will see a window showing the car driving in real-time!")
            print("This is slower than normal training but great for watching progress.")
            print("Recommended: --n-envs 1 for best live rendering experience")
            print("="*70 + "\n")
        
        TRAINING_CONFIG = {
            "total_timesteps": args.total_timesteps,
            "learning_rate": 3e-4,
            "buffer_size": 1000_000,
            "learning_starts": 1000,
            "batch_size": 256,
            "n_envs": args.n_envs,
            "save_dir": "models/sac_metadrive",
            "save_freq": 10000,
            "visualize": args.visualize,
            "live_render": args.live_render,
            "record_video": args.record_video,
            "video_interval": args.video_interval,
            "plot_interval": args.plot_interval,
            "device": args.device,
            "use_bev": args.bev,
        }
        
        trained_model = train_sac_agent(**TRAINING_CONFIG)
        
        print("\n\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("To test the trained agent with live rendering:")
        print(f"  uv run src/sac_training/train.py --test models/sac_metadrive/sac_metadrive_final")
        if args.record_video:
            print(f"\nVideos saved in: models/sac_metadrive/videos/")
        print("="*70)