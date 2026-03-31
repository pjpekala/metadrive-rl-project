"""Training callbacks for SAC agent training.

This module provides custom callbacks for stable-baselines3 SAC training:
- VideoRecorderCallback: Records videos of the agent during training
- VisualizationCallback: Real-time visualization of training progress

Usage:
    >>> from src.sac_training.callbacks import VideoRecorderCallback, VisualizationCallback
    >>> from stable_baselines3 import SAC
    >>>
    >>> model = SAC(...)
    >>> callbacks = [
    ...     VideoRecorderCallback(eval_env, "videos/"),
    ...     VisualizationCallback(plot_interval=100),
    ... ]
    >>> model.learn(total_timesteps=100000, callback=callbacks)
"""

from pathlib import Path
from typing import Optional, List, Any

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback


class VideoRecorderCallback(BaseCallback):
    """Callback for recording videos of the agent during training.
    
    Records videos at specified intervals and saves best episodes based on
    reward. Requires imageio to be installed.
    
    Attributes:
        eval_env: Environment for video recording (should have render enabled)
        video_folder: Path to save videos
        record_interval: Record video every N episodes
        video_length: Maximum steps per video
        episode_count: Current episode count
        best_reward: Best reward seen so far
    
    Example:
        >>> from src.sac_training.callbacks import VideoRecorderCallback
        >>> 
        >>> video_callback = VideoRecorderCallback(
        ...     eval_env=env,
        ...     video_folder="videos/",
        ...     record_interval=50,
        ...     video_length=500,
        ...     verbose=1,
        ... )
        >>> model.learn(total_timesteps=100000, callback=video_callback)
    """
    
    def __init__(
        self,
        eval_env: Any,
        video_folder: str,
        record_interval: int = 50,
        video_length: int = 500,
        verbose: int = 0,
    ) -> None:
        """Initialize the video recorder callback.
        
        Args:
            eval_env: Environment for video recording (must support render)
            video_folder: Directory path to save videos
            record_interval: Record video every N episodes
            video_length: Maximum steps per video
            verbose: Verbosity level (0 = silent, 1 = print messages)
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.video_folder = Path(video_folder)
        self.video_folder.mkdir(parents=True, exist_ok=True)
        self.record_interval = record_interval
        self.video_length = video_length
        self.episode_count: int = 0
        self.best_reward: float = -float("inf")
    
    def _on_step(self) -> bool:
        """Called at each step. Records video if episode ended at interval.
        
        Returns:
            True to continue training
        """
        # Check if episode ended
        if self.locals["dones"][0]:
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
    
    def _record_video(
        self,
        deterministic: bool = True,
        prefix: str = "episode",
    ) -> float:
        """Record a single episode as video.
        
        Args:
            deterministic: Whether to use deterministic actions
            prefix: Filename prefix for the video
            
        Returns:
            Total reward accumulated during the episode
        """
        if not IMAGEIO_AVAILABLE:
            return 0.0
        
        frames: List[np.ndarray] = []
        obs, info = self.eval_env.reset()
        done = False
        total_reward: float = 0.0
        steps: int = 0
        
        while not done and steps < self.video_length:
            # Render frame
            frame = self.eval_env.render()
            if frame is not None:
                frames.append(frame)
            
            # Take action
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            steps += 1
        
        # Save video
        if frames:
            video_path = self.video_folder / f"{prefix}_ep{self.episode_count}_r{total_reward:.0f}.mp4"
            imageio.mimsave(str(video_path), frames, fps=30)
            if self.verbose > 0:
                print(f"📹 Video saved: {video_path} (Reward: {total_reward:.2f})")
        
        return total_reward


class VisualizationCallback(BaseCallback):
    """Callback for visualizing training progress in real-time.
    
    Creates live plots showing episode rewards, lengths, and success rates.
    Updates periodically during training. Requires matplotlib.
    
    Attributes:
        render_env: Optional environment for rendering (not currently used)
        render_interval: Interval for rendering (not currently used)
        plot_interval: Update plots every N steps
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        episode_successes: List of episode success flags
        current_episode_reward: Reward for current episode
        current_episode_length: Length of current episode
        episode_count: Total episodes completed
        fig: Matplotlib figure
        axes: Matplotlib axes
    
    Example:
        >>> from src.sac_training.callbacks import VisualizationCallback
        >>> 
        >>> viz_callback = VisualizationCallback(
        ...     plot_interval=100,
        ...     verbose=1,
        ... )
        >>> model.learn(total_timesteps=100000, callback=viz_callback)
    """
    
    def __init__(
        self,
        render_env: Optional[Any] = None,
        render_interval: int = 10,
        plot_interval: int = 100,
        verbose: int = 0,
    ) -> None:
        """Initialize the visualization callback.
        
        Args:
            render_env: Optional environment for rendering (not used)
            render_interval: Interval for rendering (not used)
            plot_interval: Update plots every N steps
            verbose: Verbosity level (0 = silent, 1 = print messages)
        """
        super().__init__(verbose)
        self.render_env = render_env
        self.render_interval = render_interval
        self.plot_interval = plot_interval
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_successes: List[float] = []
        self.current_episode_reward: float = 0.0
        self.current_episode_length: int = 0
        self.episode_count: int = 0
        self.fig: Optional[Any] = None
        self.axes: Optional[Any] = None
    
    def _on_training_start(self) -> None:
        """Called at the start of training."""
        if self.verbose > 0:
            print("\n[Visualization] Starting training visualization...")
    
    def _on_step(self) -> bool:
        """Called at each step. Updates tracking and plots periodically.
        
        Returns:
            True to continue training
        """
        # Track episode statistics
        self.current_episode_reward += float(self.locals["rewards"][0])
        self.current_episode_length += 1
        
        # Check if episode ended
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Check if agent succeeded
            info = self.locals.get("infos", [{}])[0]
            success = info.get("arrive_dest", False)
            self.episode_successes.append(1.0 if success else 0.0)
            
            self.episode_count += 1
            
            if self.verbose > 0:
                success_rate = (
                    np.mean(self.episode_successes[-100:]) * 100
                    if self.episode_successes
                    else 0
                )
                avg_reward = (
                    np.mean(self.episode_rewards[-100:])
                    if self.episode_rewards
                    else 0
                )
                print(
                    f"\n[Episode {self.episode_count}] "
                    f"Reward: {self.current_episode_reward:.2f} | "
                    f"Length: {self.current_episode_length} | "
                    f"Success: {'✓' if success else '✗'} | "
                    f"Avg Reward (100): {avg_reward:.2f} | "
                    f"Success Rate (100): {success_rate:.1f}%"
                )
            
            # Reset episode tracking
            self.current_episode_reward = 0.0
            self.current_episode_length = 0
        
        # Update plots periodically
        if self.num_timesteps % self.plot_interval == 0 and len(self.episode_rewards) > 0:
            self._update_plots()
        
        return True
    
    def _update_plots(self) -> None:
        """Update live training plots with current statistics."""
        if len(self.episode_rewards) < 2:
            return
        
        if self.fig is None:
            plt.ion()
            self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
            self.fig.suptitle("MetaDrive Training Progress", fontsize=16)
        
        # Clear all subplots
        for ax in self.axes.flat:
            ax.clear()
        
        episodes = np.arange(len(self.episode_rewards))
        
        # Plot 1: Episode Rewards
        ax1 = self.axes[0, 0]
        ax1.plot(episodes, self.episode_rewards, alpha=0.3, label="Raw")
        if len(self.episode_rewards) >= 10:
            smooth_rewards = np.convolve(
                self.episode_rewards, np.ones(10) / 10, mode="valid"
            )
            ax1.plot(episodes[9:], smooth_rewards, linewidth=2, label="Smoothed (10)")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.set_title("Episode Rewards")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Episode Lengths
        ax2 = self.axes[0, 1]
        ax2.plot(episodes, self.episode_lengths, alpha=0.3, label="Raw")
        if len(self.episode_lengths) >= 10:
            smooth_lengths = np.convolve(
                self.episode_lengths, np.ones(10) / 10, mode="valid"
            )
            ax2.plot(episodes[9:], smooth_lengths, linewidth=2, label="Smoothed (10)")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Episode Length")
        ax2.set_title("Episode Lengths")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Success Rate
        ax3 = self.axes[1, 0]
        if len(self.episode_successes) >= 10:
            window = min(100, len(self.episode_successes))
            success_rates = [
                np.mean(self.episode_successes[max(0, i - window) : i + 1]) * 100
                for i in range(len(self.episode_successes))
            ]
            ax3.plot(episodes, success_rates, linewidth=2)
            ax3.set_xlabel("Episode")
            ax3.set_ylabel("Success Rate (%)")
            ax3.set_title(f"Success Rate (rolling window={window})")
            ax3.set_ylim([0, 105])
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Statistics Summary
        ax4 = self.axes[1, 1]
        ax4.axis("off")
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
        ax4.text(
            0.1,
            0.5,
            stats_text,
            fontsize=11,
            family="monospace",
            verticalalignment="center",
        )
        
        plt.tight_layout()
        plt.pause(0.001)
