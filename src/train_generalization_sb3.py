"""
Modified training script using Stable Baselines3 + Ray Tune instead of RLlib.
Drops the gym wrapper requirement entirely, using MetaDriveEnv directly with gymnasium.

Dependencies:
    stable-baselines3[extra]>=2.0.0
    torch
    gymnasium
    metadrive-simulator (from git)
    ray[tune]>=2.54.0
"""

import argparse
import os
import logging
from typing import Optional

import numpy as np
import torch

from metadrive import MetaDriveEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

import ray
from ray import tune
from ray.tune import CLIReporter


# ──────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────

class MetaDriveMetricsCallback(BaseCallback):
    """Collects per-episode MetaDrive info metrics and reports them to Ray Tune."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_metrics = []

    def _on_step(self) -> bool:
        # SB3 Monitor wrapper writes episode info into the info dict under "episode"
        for info in self.locals.get("infos", []):
            if "arrive_dest" in info:
                self._episode_metrics.append({
                    "success":    float(info.get("arrive_dest", False)),
                    "crash":      float(info.get("crash", False)),
                    "out":        float(info.get("out_of_road", False)),
                    "cost":       float(info.get("cost", 0.0)),
                    "velocity":   float(info.get("velocity", 0.0)),
                    "step_reward": float(info.get("step_reward", 0.0)),
                })
        return True

    def get_mean(self, key: str) -> Optional[float]:
        if not self._episode_metrics:
            return float("nan")
        vals = [m[key] for m in self._episode_metrics if key in m]
        return float(np.mean(vals)) if vals else float("nan")

    def reset_metrics(self):
        self._episode_metrics = []


# ──────────────────────────────────────────────
# Environment factory helpers
# ──────────────────────────────────────────────

def make_env(env_config: dict, rank: int = 0, seed: int = 0):
    """Returns a callable that creates a single monitored MetaDriveEnv."""
    def _init():
        cfg = dict(env_config)
        cfg["start_seed"] = cfg.get("start_seed", seed) + rank
        env = MetaDriveEnv(cfg)
        env = Monitor(env)
        return env
    return _init


def make_eval_env(eval_env_config: dict, n_envs: int = 5):
    return SubprocVecEnv([make_env(eval_env_config, rank=i) for i in range(n_envs)])


# ──────────────────────────────────────────────
# Single training run (called by Ray Tune)
# ──────────────────────────────────────────────

def run_trial(config: dict):
    """One trial: train a PPO agent and periodically report metrics to Tune."""

    # ── Build training vec-env ──────────────────────────────────────────
    train_env_config = {
        "num_scenarios":  config["num_scenarios"],
        "start_seed":     config["start_seed"],
        "random_traffic": False,
        "traffic_density": config["traffic_density"],
    }
    eval_env_config = {
        "num_scenarios": 200,
        "start_seed":    0,
        "random_traffic": False,
        "traffic_density": config["traffic_density"],
    }

    n_workers = config.get("num_workers", 5)
    train_vec_env = SubprocVecEnv(
        [make_env(train_env_config, rank=i) for i in range(n_workers)]
    )
    eval_vec_env = make_eval_env(eval_env_config, n_envs=config.get("eval_num_workers", 5))

    # ── Build PPO model ────────────────────────────────────────────────
    model = PPO(
        policy="MlpPolicy",
        env=train_vec_env,
        learning_rate=config.get("lr", 3e-4),
        n_steps=config.get("rollout_fragment_length", 200),        # steps per worker per update
        batch_size=config.get("sgd_minibatch_size", 256),
        n_epochs=config.get("num_sgd_iter", 10),
        gae_lambda=config.get("lambda", 0.95),
        device="cuda" if torch.cuda.is_available() and config.get("use_gpu", False) else "cpu",
        verbose=0,
    )

    # ── Training loop ──────────────────────────────────────────────────
    total_timesteps    = config.get("total_timesteps", 10_000_000)
    train_batch_size   = config.get("train_batch_size", 20_000)
    evaluation_interval = config.get("evaluation_interval", 2)   # every N batches
    eval_episodes      = config.get("evaluation_num_episodes", 40)

    metrics_cb = MetaDriveMetricsCallback()
    timesteps_done = 0
    batch_idx = 0

    # Save checkpoints to the experiment output dir, not /tmp, to avoid filling Ray's session dir
    checkpoint_dir = os.path.join(os.path.abspath("."), "checkpoints", os.environ.get("TUNE_TRIAL_NAME", "trial"))
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_freq = config.get("checkpoint_freq", 10)
    report = {}

    while timesteps_done < total_timesteps:
        # ── Learn for one batch ────────────────────────────────────────
        model.learn(
            total_timesteps=train_batch_size,
            callback=metrics_cb,
            reset_num_timesteps=False,
        )
        timesteps_done += train_batch_size
        batch_idx += 1

        report = {
            "timesteps_total": timesteps_done,
            "success": metrics_cb.get_mean("success"),
            "crash":   metrics_cb.get_mean("crash"),
            "out":     metrics_cb.get_mean("out"),
            "cost":    metrics_cb.get_mean("cost"),
        }
        metrics_cb.reset_metrics()

        # ── Evaluation ────────────────────────────────────────────────
        if batch_idx % evaluation_interval == 0:
            eval_successes, eval_crashes, eval_outs, eval_lengths = [], [], [], []
            obs = eval_vec_env.reset()
            ep_lengths = np.zeros(eval_vec_env.num_envs, dtype=int)
            completed = 0

            while completed < eval_episodes:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = eval_vec_env.step(action)
                ep_lengths += 1

                for i, (done, info) in enumerate(zip(dones, infos)):
                    if done:
                        completed += 1
                        eval_successes.append(float(info.get("arrive_dest", False)))
                        eval_crashes.append(float(info.get("crash", False)))
                        eval_outs.append(float(info.get("out_of_road", False)))
                        eval_lengths.append(ep_lengths[i])
                        ep_lengths[i] = 0
                        if completed >= eval_episodes:
                            break

            report["eval/success_rate"]     = float(np.mean(eval_successes))
            report["eval/crash_rate"]       = float(np.mean(eval_crashes))
            report["eval/out_of_road_rate"] = float(np.mean(eval_outs))
            report["eval/episode_len_mean"] = float(np.mean(eval_lengths))

        # ── Checkpoint ────────────────────────────────────────────────
        if batch_idx % checkpoint_freq == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"model_step_{timesteps_done}.zip")
            model.save(ckpt_path)

        # ── Report to Tune every batch ─────────────────────────────────
        tune.report(metrics=report)

    # ── Final checkpoint ──────────────────────────────────────────────
    model.save(os.path.join(checkpoint_dir, f"model_final_{timesteps_done}.zip"))
    tune.report(metrics=report)

    train_vec_env.close()
    eval_vec_env.close()


# ──────────────────────────────────────────────
# CLI + main
# ──────────────────────────────────────────────

def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="generalization_experiment")
    parser.add_argument("--num-gpus", type=int, default=0)
    return parser


if __name__ == "__main__":
    args = get_train_parser().parse_args()

    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    ray.init(logging_level=logging.ERROR)

    # Mirror the original grid search over num_scenarios and traffic_density.
    # start_seed kept as a grid_search list to match original intent (only 5000 here).
    search_space = {
        "num_scenarios":   tune.grid_search([1, 3, 5, 1000]),
        "start_seed":      tune.grid_search([5000]),
        "traffic_density": tune.grid_search([0.1, 0.3]),

        # ── PPO hyper-parameters ────────────────────────────────────
        "lr":                      3e-4,
        "rollout_fragment_length": 200,
        "sgd_minibatch_size":      256,
        "train_batch_size":        20_000,
        "num_sgd_iter":            10,
        "lambda":                  0.95,

        # ── Infrastructure ──────────────────────────────────────────
        "num_workers":           5,
        "eval_num_workers":      5,
        "evaluation_interval":   2,
        "evaluation_num_episodes": 40,
        "total_timesteps":       10_000_000,
        "checkpoint_freq":       10,
        "use_gpu":               args.num_gpus > 0,
    }

    progress_reporter = CLIReporter(
        metric_columns={
            "timesteps_total":       "timesteps",
            "success":               "success",
            "crash":                 "crash",
            "out":                   "out",
            "cost":                  "cost",
            "eval/success_rate":     "eval_success",
            "eval/episode_len_mean": "eval_len",
            "training_iteration":    "iter",
        }
    )

    tune.run(
        run_trial,
        name=args.exp_name,
        config=search_space,
        stop={"timesteps_total": 10_000_000},
        max_failures=5,
        reuse_actors=False,
        storage_path=os.path.abspath("."),
        verbose=1,
        progress_reporter=progress_reporter,
        resources_per_trial={
            "cpu": 6,
            "gpu": 0.25 if args.num_gpus > 0 else 0,
        },
    )