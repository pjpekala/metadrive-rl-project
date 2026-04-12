"""
MetaDrive RL Evaluation Script
==============================
Features:
- Evaluate trained SAC models on held-out seeds (800-999)
- Deterministic policy execution for reproducible results
- Video recording of evaluation episodes
- Statistical summary of success, collision, and off-road rates

Usage:
    # Basic evaluation (headless)
    python evaluate_metadrive.py --model-path ./models/bev/sac_metadrive_final.zip --modality bev

    # Evaluate with rendering
    python evaluate_metadrive.py --model-path ./models/bev/sac_metadrive_final.zip --modality bev --render

    # Evaluate with video recording
    python evaluate_metadrive.py --model-path ./models/rgb/sac_metadrive_final.zip --modality rgb --save-video --video-dir ./videos

    # Evaluate with specific number of episodes
    python evaluate_metadrive.py --model-path ./models/bev/sac_metadrive_final.zip --modality bev --episodes 100

    # Quick sanity-check eval (fast)
    python evaluate_metadrive.py --model-path ./models/bev/sac_metadrive_final.zip --modality bev --episodes 10 --no-save-csv
"""

import os
import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np

from stable_baselines3 import SAC

from train_metadrive import (
    MODALITY_CONFIG,
    EVAL_SEED_START,
    EVAL_SEED_END,
    create_metadrive_env,
    get_env_config,
)

EVAL_RESULTS_HEADER = [
    "modality",
    "model_path",
    "episodes",
    "success_rate",
    "collision_rate",
    "out_of_road_rate",
    "mean_reward",
    "mean_episode_length",
    "mean_speed",
    "std_reward",
    "std_episode_length",
]

EVAL_RESULTS_CSV = "evaluation_results.csv"


def evaluate_model(
    model_path: str,
    modality: str,
    num_episodes: int = 50,
    render: bool = False,
    save_video: bool = False,
    video_dir: str = "./videos",
) -> dict:
    modality_config = MODALITY_CONFIG.get(modality, MODALITY_CONFIG["sparse_lidar"])
    policy_type = modality_config["policy"]

    print("=" * 70)
    print(f"MetaDrive Evaluation - {modality_config['description']}")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Policy: {policy_type}")
    print(f"Seeds: {EVAL_SEED_START}-{EVAL_SEED_END} (held-out)")
    print(f"Episodes: {num_episodes}")
    print(f"Render: {'Yes' if render else 'No'}")
    print(f"Save Video: {'Yes' if save_video else 'No'}")
    print("=" * 70)

    env = create_metadrive_env(modality, is_eval=True, render=render)
    model = SAC.load(model_path, env=env)
    print(f"[+] Model loaded from: {model_path}")
    print(f"[+] Observation space: {env.observation_space}")
    print(f"[+] Action space: {env.action_space}")

    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    episode_collisions = []
    episode_out_of_road = []
    episode_speeds = []

    video_frames = [] if save_video else None
    video_writer = None

    if save_video:
        video_dir = Path(video_dir) / modality
        video_dir.mkdir(parents=True, exist_ok=True)
        try:
            import imageio
            video_path = video_dir / f"eval_{Path(model_path).stem}.mp4"
            video_writer = imageio.get_writer(str(video_path))
            print(f"[+] Video will be saved to: {video_path}")
        except ImportError:
            print("[!] imageio not available. Video recording disabled.")
            save_video = False

    print("\nStarting evaluation...")
    print("-" * 70)

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_speed = 0
        done = False

        print(f"\n[Episode {ep + 1}/{num_episodes}] ", end="")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            if render and episode_length % 10 == 0:
                frame = env.render()
                if frame is not None and save_video and video_writer is not None:
                    video_writer.append_data(frame)

        success = info.get("arrive_dest", False)
        collision = info.get("crash", False)
        out_of_road = info.get("out_of_road", False)

        try:
            avg_speed = episode_reward / max(episode_length, 1)
        except Exception:
            avg_speed = 0

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_successes.append(float(success))
        episode_collisions.append(float(collision))
        episode_out_of_road.append(float(out_of_road))
        episode_speeds.append(avg_speed)

        status = "SUCCESS" if success else ("CRASH" if collision else ("OUT_OF_ROAD" if out_of_road else "TIMEOUT"))
        print(f"{status} | Reward: {episode_reward:.2f} | Length: {episode_length}")

    if save_video and video_writer is not None:
        video_writer.close()
        print(f"\n[+] Video saved to: {video_path}")

    env.close()

    results = {
        "modality": modality,
        "model_path": model_path,
        "episodes": num_episodes,
        "success_rate": np.mean(episode_successes),
        "collision_rate": np.mean(episode_collisions),
        "out_of_road_rate": np.mean(episode_out_of_road),
        "mean_reward": np.mean(episode_rewards),
        "mean_episode_length": np.mean(episode_lengths),
        "mean_speed": np.mean(episode_speeds),
        "std_reward": np.std(episode_rewards),
        "std_episode_length": np.std(episode_lengths),
    }

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Success Rate:       {results['success_rate']*100:.1f}%")
    print(f"Collision Rate:     {results['collision_rate']*100:.1f}%")
    print(f"Out-of-Road Rate:   {results['out_of_road_rate']*100:.1f}%")
    print(f"Mean Reward:       {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"Mean Episode Len:   {results['mean_episode_length']:.1f} +/- {results['std_episode_length']:.1f}")
    print(f"Mean Speed:        {results['mean_speed']:.2f}")
    print("=" * 70)

    return results


def save_results(results: dict, csv_path: str = EVAL_RESULTS_CSV):
    csv_path = Path(csv_path)
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_RESULTS_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)

    print(f"[+] Results appended to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained SAC model on MetaDrive held-out maps"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model (.zip file)",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="sparse_lidar",
        choices=list(MODALITY_CONFIG.keys()),
        help="Sensor modality (for observation space config)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render evaluation episodes",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save video of evaluation episodes",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="./videos",
        help="Directory to save videos",
    )
    parser.add_argument(
        "--save-csv",
        dest="save_csv",
        type=lambda x: x.lower() != "false",
        default=True,
        help="Save results to CSV (default: enabled, use --no-save-csv to disable)",
    )
    parser.add_argument(
        "--no-save-csv",
        dest="save_csv",
        action="store_false",
        help="Disable saving results to CSV",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    results = evaluate_model(
        model_path=args.model_path,
        modality=args.modality,
        num_episodes=args.episodes,
        render=args.render,
        save_video=args.save_video,
        video_dir=args.video_dir,
    )

    if args.save_csv:
        save_results(results)
 
