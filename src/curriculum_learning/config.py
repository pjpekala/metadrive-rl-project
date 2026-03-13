"""
Configuration module for curriculum learning.

This is the SINGLE SOURCE of truth for all stage definitions and training config.
All other modules import from here.
"""

from dataclasses import dataclass, field
from typing import Union, Optional


@dataclass
class Stage:
    """
    Single curriculum stage configuration.
    
    Training and evaluation both use this config - ensuring alignment.
    """
    name: str
    map: Union[int, str]
    num_scenarios_train: int
    num_scenarios_eval: int
    traffic_density: float = 0.0
    
    min_success_rate: float = 0.8
    min_survival_rate: float = 0.95
    reward_threshold: float = 200.0
    
    discrete_action: Optional[bool] = None
    discrete_steering_dim: int = 3
    discrete_throttle_dim: int = 3
    horizon: int = 1000
    
    @property
    def is_discrete(self) -> bool:
        return self.discrete_action if self.discrete_action is not None else False


@dataclass
class Seeds:
    """Seed configuration for reproducibility"""
    train: int = 0
    inline_eval: int = 5000
    held_out_eval: int = 2000


@dataclass
class TrainingConfig:
    """Main training hyperparameters"""
    n_envs: int = 4
    n_steps: int = 2048
    batch_size: int = 512
    discrete_timesteps: int = 10_000_000
    continuous_timesteps: int = 50_000_000
    eval_freq_steps: int = 100_000
    n_eval_episodes: int = 20
    device: str = "cpu"
    seeds: Seeds = field(default_factory=Seeds)
    
    policy: str = "MlpPolicy"
    net_arch: dict = field(default_factory=lambda: dict(pi=[256, 256], vf=[256, 256]))
    
    learning_rate: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    n_epochs: int = 10


CONTINUOUS_CURRICULUM = [
    Stage(
        name="Stage 1 — Straight road (S), no traffic",
        map="S",
        num_scenarios_train=20,
        num_scenarios_eval=20,
        traffic_density=0.0,
        min_success_rate=0.85,
        min_survival_rate=0.90,
        reward_threshold=95,
        discrete_action=False,
    ),
    Stage(
        name="Stage 2 — Straight roads (SSS), no traffic",
        map="SSS",
        num_scenarios_train=30,
        num_scenarios_eval=30,
        traffic_density=0.0,
        min_success_rate=0.65,
        min_survival_rate=0.80,
        reward_threshold=100,
        discrete_action=False,
    ),
    Stage(
        name="Stage 3 — Straight roads (SSS), light traffic",
        map="SSS",
        num_scenarios_train=40,
        num_scenarios_eval=40,
        traffic_density=0.05,
        min_success_rate=0.65,
        min_survival_rate=0.80,
        reward_threshold=100,
        discrete_action=False,
    ),
    Stage(
        name="Stage 4 — Curves (SCC), no traffic",
        map="SCC",
        num_scenarios_train=50,
        num_scenarios_eval=50,
        traffic_density=0.0,
        min_success_rate=0.60,
        min_survival_rate=0.80,
        reward_threshold=100,
        discrete_action=False,
    ),
    Stage(
        name="Stage 5 — Curves (SCC), light traffic",
        map="SCC",
        num_scenarios_train=75,
        num_scenarios_eval=75,
        traffic_density=0.05,
        min_success_rate=0.60,
        min_survival_rate=0.75,
        reward_threshold=100,
        discrete_action=False,
    ),
    Stage(
        name="Stage 6 — Mixed (XCSO), light traffic",
        map="XCSO",
        num_scenarios_train=100,
        num_scenarios_eval=100,
        traffic_density=0.05,
        min_success_rate=0.55,
        min_survival_rate=0.70,
        reward_threshold=100,
        discrete_action=False,
    ),
    Stage(
        name="Stage 7 — Mixed (XCSO), moderate traffic",
        map="XCSO",
        num_scenarios_train=150,
        num_scenarios_eval=150,
        traffic_density=0.1,
        min_success_rate=0.50,
        min_survival_rate=0.70,
        reward_threshold=100,
        discrete_action=False,
    ),
    Stage(
        name="Stage 8 — Random maps (5 blocks), moderate traffic",
        map=5,
        num_scenarios_train=200,
        num_scenarios_eval=200,
        traffic_density=0.15,
        min_success_rate=0.50,
        min_survival_rate=0.70,
        reward_threshold=125,
        discrete_action=False,
    ),
    Stage(
        name="Stage 9 — Random maps (6 blocks), moderate traffic",
        map=6,
        num_scenarios_train=300,
        num_scenarios_eval=300,
        traffic_density=0.25,
        min_success_rate=0.50,
        min_survival_rate=0.70,
        reward_threshold=125,
        discrete_action=False,
    ),
    Stage(
        name="Stage 10 — Random maps (6 blocks), heavy traffic",
        map=6,
        num_scenarios_train=400,
        num_scenarios_eval=400,
        traffic_density=0.35,
        min_success_rate=0.50,
        min_survival_rate=0.70,
        reward_threshold=130,
        discrete_action=False,
    ),
    Stage(
        name="Stage 11 — Random maps (7 blocks), heavy traffic",
        map=7,
        num_scenarios_train=500,
        num_scenarios_eval=500,
        traffic_density=0.40,
        min_success_rate=0.45,
        min_survival_rate=0.65,
        reward_threshold=140,
        discrete_action=False,
    ),
    Stage(
        name="Stage 12 — Random maps (9 blocks), heavy traffic (final)",
        map=9,
        num_scenarios_train=750,
        num_scenarios_eval=750,
        traffic_density=0.7,
        min_success_rate=0.0,
        min_survival_rate=0.0,
        reward_threshold=float("inf"),
        discrete_action=False,
        horizon=1200,
    ),
]


DISCRETE_CURRICULUM = [
    Stage(
        name="Stage 0 — Straight (S), no traffic",
        map="S",
        num_scenarios_train=5,
        num_scenarios_eval=5,
        traffic_density=0.0,
        min_success_rate=1.0,
        min_survival_rate=1.0,
        reward_threshold=100,
        discrete_action=True,
        discrete_steering_dim=3,
        discrete_throttle_dim=3,
    ),
    Stage(
        name="Stage 1 — Curves (SC), no traffic",
        map="SC",
        num_scenarios_train=10,
        num_scenarios_eval=10,
        traffic_density=0.0,
        min_success_rate=1.0,
        min_survival_rate=1.0,
        reward_threshold=110,
        discrete_action=True,
        discrete_steering_dim=5,
        discrete_throttle_dim=3,
    ),
    Stage(
        name="Stage 2 — Mixed (XCS), light traffic",
        map="XCS",
        num_scenarios_train=20,
        num_scenarios_eval=20,
        traffic_density=0.05,
        min_success_rate=0.75,
        min_survival_rate=0.90,
        reward_threshold=110,
        discrete_action=True,
        discrete_steering_dim=5,
        discrete_throttle_dim=3,
    ),
    Stage(
        name="Stage 3 — Full (XCSO), moderate traffic",
        map="XCSO",
        num_scenarios_train=50,
        num_scenarios_eval=50,
        traffic_density=0.1,
        min_success_rate=0.0,
        min_survival_rate=0.0,
        reward_threshold=float("inf"),
        discrete_action=True,
        discrete_steering_dim=5,
        discrete_throttle_dim=3,
    ),
]


CURRICULA = {
    "discrete": DISCRETE_CURRICULUM,
    "continuous": CONTINUOUS_CURRICULUM,
}
