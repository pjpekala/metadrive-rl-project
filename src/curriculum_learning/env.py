"""
Environment factories and wrappers.

All environment configuration goes through build_metadrive_config() to ensure
training and evaluation use the same settings.
"""

from typing import Optional, List
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from stable_baselines3.common.monitor import Monitor

from .config import Stage


def build_metadrive_config(
    stage: Stage,
    seed: int,
    scenario_index: int = 0,
    use_render: bool = False,
    for_training: bool = True,
) -> dict:
    """
    Build a MetaDrive config dict for a given curriculum stage.
    
    This is the SINGLE SOURCE OF TRUTH for environment configuration.
    Used by training env, inline eval env, and standalone evaluation.
    
    Args:
        stage: Curriculum stage configuration
        seed: Base seed for this environment
        scenario_index: Worker index; offsets seed for different scenarios per worker
        use_render: Enable rendering (for visualization)
        for_training: If True, use num_scenarios_train; else use num_scenarios_eval
    """
    num_scenarios = stage.num_scenarios_train if for_training else stage.num_scenarios_eval
    
    config = {
        "num_scenarios": num_scenarios,
        "start_seed": seed + scenario_index * num_scenarios,
        "map": stage.map,
        "traffic_density": stage.traffic_density,
        
<<<<<<< Updated upstream
        "driving_reward": 1.0,
        "speed_reward": 0.2,
        "use_lateral_reward": True,
        "success_reward": 30.0,
        "out_of_road_penalty": 15.0,
        "crash_vehicle_penalty": 15.0,
=======
        "success_reward": 30.0,
        "out_of_road_penalty": 30.0,
        "crash_vehicle_penalty": 20.0,
        
        "use_lateral_reward": True,
        "driving_reward": 1.5,
        "speed_reward": 0.1,
>>>>>>> Stashed changes
        
        "out_of_road_done": True,
        "crash_vehicle_done": True,
        "crash_object_done": True,
        
        "horizon": stage.horizon,
        "use_render": use_render,
        "log_level": 50,
        
        "discrete_action": stage.is_discrete,
        
        "random_lane_width": stage.random_lane_width,
        "random_lane_num": stage.random_lane_num,
        "random_agent_model": stage.random_agent_model,
        "random_traffic": stage.random_traffic,
    }
    
    if stage.is_discrete:
        config["discrete_steering_dim"] = stage.discrete_steering_dim
        config["discrete_throttle_dim"] = stage.discrete_throttle_dim
    
    return config


class CurriculumEnv(SafeMetaDriveEnv):
    """
    Training environment that supports live curriculum stage updates.
    
    Each worker holds a reference to a shared `stage_holder` list.
    On reset(), if the stage has changed, the env reinitializes its config.
    """
    def __init__(
        self,
        config: Optional[dict],
        train_seed: int,
        scenario_index: int,
        stage_holder: List[Stage],
    ):
        self.train_seed = train_seed
        self.scenario_index = scenario_index
        self.stage_holder = stage_holder
        self._current_stage_name = None
        
        initial_config = build_metadrive_config(
            stage_holder[0], train_seed, scenario_index, for_training=True
        )
        super().__init__(config=initial_config)
    
    def reset(self, **kwargs):
        kwargs.pop("seed", None)
        kwargs.pop("options", None)
        
        stage = self.stage_holder[0]
        if stage.name != self._current_stage_name:
            self._current_stage_name = stage.name
            self.config.update(
                build_metadrive_config(stage, self.train_seed, self.scenario_index, for_training=True)
            )
        return super().reset(**kwargs)
    
    def step(self, action):
        try:
            return super().step(action)
        except Exception as e:
            print(f"[CurriculumEnv] step() error: {e} — treating as termination")
            obs = super().reset()[0]
            return obs, -10.0, True, False, {}


class InlineEvalEnv(SafeMetaDriveEnv):
    """
    Eval environment that tracks curriculum stage changes.
    
    Mirrors CurriculumEnv's stage-tracking so inline EvalCallback
    always evaluates on the same stage the agent is currently training on.
    """
    def __init__(self, eval_seed: int, stage_holder: List[Stage]):
        self.eval_seed = eval_seed
        self.stage_holder = stage_holder
        self._current_stage_name = None
        
        initial_config = build_metadrive_config(
            stage_holder[0], eval_seed, for_training=False
        )
        super().__init__(config=initial_config)
    
    def reset(self, **kwargs):
        kwargs.pop("seed", None)
        kwargs.pop("options", None)
        
        stage = self.stage_holder[0]
        if stage.name != self._current_stage_name:
            self._current_stage_name = stage.name
            self.config.update(
                build_metadrive_config(stage, self.eval_seed, for_training=False)
            )
        return super().reset(**kwargs)
    
    def step(self, action):
        try:
            return super().step(action)
        except Exception as e:
            print(f"[InlineEvalEnv] step() error: {e} — treating as termination")
            obs = super().reset()[0]
            return obs, -10.0, True, False, {}


class HeldOutEvalEnv(SafeMetaDriveEnv):
    """
    Standalone evaluation environment for held-out scenarios.
    
    Used by evaluate.py for final model evaluation on unseen scenarios.
    """
    def __init__(self, config: dict):
        super().__init__(config=config)
        
        original_reset = self.reset
        def reset_no_seed(**kwargs):
            kwargs.pop("seed", None)
            kwargs.pop("options", None)
            return original_reset(**kwargs)
        self.reset = reset_no_seed


def make_env(
    train_seed: int,
    scenario_index: int,
    stage_holder: List[Stage],
) -> callable:
    """Factory for training environment."""
    def _init():
        env = CurriculumEnv(
            config=None,
            train_seed=train_seed,
            scenario_index=scenario_index,
            stage_holder=stage_holder,
        )
        env = Monitor(env)
        return env
    return _init


def make_inline_eval_env(
    eval_seed: int,
    stage_holder: List[Stage],
) -> callable:
    """Factory for inline evaluation environment."""
    def _init():
        env = InlineEvalEnv(eval_seed=eval_seed, stage_holder=stage_holder)
        env = Monitor(env)
        return env
    return _init


def make_held_out_eval_env(
    seed: int,
    stage: Stage,
    use_render: bool = False,
) -> callable:
    """Factory for held-out evaluation environment."""
    def _init():
        config = build_metadrive_config(
            stage=stage,
            seed=seed,
            scenario_index=0,
            use_render=use_render,
            for_training=False,
        )
        env = HeldOutEvalEnv(config=config)
        env = Monitor(env)
        return env
    return _init
