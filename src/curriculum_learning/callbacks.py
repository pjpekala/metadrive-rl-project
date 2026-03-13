"""
Training callbacks for curriculum learning.

- CurriculumCallback: tracks stage progress and syncs VecNormalize
- CurriculumEvalCallback: runs evaluation and determines stage advancement
- EarlyStoppingCallback: stops training when final stage thresholds are met
"""

import os
import numpy as np
from typing import Optional, List
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback


class CurriculumCallback(BaseCallback):
    """
    Tracks curriculum stage progression and syncs VecNormalize stats.
    
    Fires on the eval schedule to sync observation normalizer stats to the
    eval environment and saves checkpoints.
    """
    def __init__(
        self,
        train_env,
        eval_env,
        eval_freq: int,
        stage_holder: List,
        curriculum: List,
        vecnorm_ckpt_path: str,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.stage_holder = stage_holder
        self.curriculum = curriculum
        self.vecnorm_ckpt_path = vecnorm_ckpt_path
        self.current_stage_idx = 0
        self.stage_start_timestep = 0
        self.eval_count = 0
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0 or self.n_calls == 1:
            self.eval_env.obs_rms = self.train_env.obs_rms
            self.eval_env.training = False
            self.train_env.save(self.vecnorm_ckpt_path)
        
        self.logger.record("curriculum/stage", self.current_stage_idx + 1)
        return True
    
    def on_eval_end(self, mean_reward: float, survival_rate: float, success_rate: float) -> bool:
        """
        Called after each evaluation round.
        Returns True if curriculum should continue, False if training should stop.
        """
        self.eval_count += 1
        stage = self.curriculum[self.current_stage_idx]
        
        reward_ok = mean_reward >= stage.reward_threshold
        survival_ok = survival_rate >= stage.min_survival_rate
        success_ok = success_rate >= stage.min_success_rate
        
        steps_in_stage = self.num_timesteps - self.stage_start_timestep
        
        print(f"\n[Curriculum] {stage.name} (eval #{self.eval_count})")
        print(f"  Steps in stage: {steps_in_stage:,}")
        print(f"  Reward:   {mean_reward:.1f} (need >= {stage.reward_threshold}) — {'✓' if reward_ok else '✗'}")
        print(f"  Survival: {survival_rate*100:.0f}% (need >= {stage.min_survival_rate*100:.0f}%) — {'✓' if survival_ok else '✗'}")
        print(f"  Success:  {success_rate*100:.0f}% (need >= {stage.min_success_rate*100:.0f}%) — {'✓' if success_ok else '✗'}")
        
        self.logger.record("curriculum/stage", self.current_stage_idx + 1)
        self.logger.record("curriculum/survival_threshold", stage.min_survival_rate)
        self.logger.record("curriculum/success_threshold", stage.min_success_rate)
        self.logger.dump(self.num_timesteps)
        
        should_advance = reward_ok and survival_ok and success_ok
        
        if should_advance and self.current_stage_idx < len(self.curriculum) - 1:
            self.current_stage_idx += 1
            new_stage = self.curriculum[self.current_stage_idx]
            self.stage_start_timestep = self.num_timesteps
            print(f"\n{'='*60}")
            print(f"[Curriculum] ✓ ADVANCING to: {new_stage.name}")
            print(f"{'='*60}\n")
            self.stage_holder[0] = new_stage
        elif should_advance and self.current_stage_idx >= len(self.curriculum) - 1:
            print(f"\n[Curriculum] ✓ Final stage '{stage.name}' thresholds met!")
        
        return should_advance


class CurriculumEvalCallback(EvalCallback):
    """
    Extended EvalCallback with detailed per-episode logging.
    
    Runs evaluation episodes, collects metrics, and notifies the
    CurriculumCallback after each evaluation round.
    """
    def __init__(
        self,
        curriculum_cb: CurriculumCallback,
        early_stopping_cb: Optional["EarlyStoppingCallback"] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.curriculum_cb = curriculum_cb
        self.early_stopping_cb = early_stopping_cb
        self._prev_mean_reward = None
    
    def evaluate_policy_verbose(self) -> tuple:
        episode_rewards, episode_lengths = [], []
        success_count = collision_count = offroad_count = failed_count = 0
        
        obs = self.eval_env.reset()
        ep_reward, ep_length, ep_num = 0.0, 0, 0
        
        print(f"\n--- Eval (step {self.num_timesteps}) ---")
        
        while ep_num < self.n_eval_episodes:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, reward, done, info = self.eval_env.step(action)
            ep_reward += reward[0]
            ep_length += 1
            
            if done[0]:
                ep_num += 1
                episode_rewards.append(ep_reward)
                episode_lengths.append(ep_length)
                
                arrive_dest = info[0].get("arrive_dest", False)
                crash = info[0].get("crash", False)
                out_of_road = info[0].get("out_of_road", False)
                
                if arrive_dest:
                    success_count += 1
                if crash:
                    collision_count += 1
                if out_of_road:
                    offroad_count += 1
                if crash or out_of_road:
                    failed_count += 1
                
                if arrive_dest:
                    status = "✓ SUCCESS"
                elif crash:
                    status = "✗ CRASH  "
                elif out_of_road:
                    status = "~ OFFROAD"
                else:
                    status = "  TIMEOUT"
                
                print(
                    f"  Ep {ep_num:2d} | {status} | "
                    f"Reward: {ep_reward:7.2f} | "
                    f"Length: {ep_length:4d}"
                )
                ep_reward, ep_length = 0.0, 0
                obs = self.eval_env.reset()
        
        n = self.n_eval_episodes
        mean_r = np.mean(episode_rewards)
        std_r = np.std(episode_rewards)
        mean_len = np.mean(episode_lengths)
        survival_rate = (n - failed_count) / n
        success_rate = success_count / n
        collision_rate = collision_count / n
        offroad_rate = offroad_count / n
        
        print(
            f"  Summary | Mean reward: {mean_r:.2f} ± {std_r:.2f} | "
            f"Success: {success_count}/{n} | "
            f"Crash: {collision_count}/{n} | "
            f"Off-road: {offroad_count}/{n} | "
            f"Survival: {int(survival_rate*n)}/{n}\n"
        )
        
        if self.logger is not None:
            self.logger.record("eval/mean_reward", mean_r)
            self.logger.record("eval/std_reward", std_r)
            self.logger.record("eval/mean_ep_length", mean_len)
            self.logger.record("eval/success_rate", success_rate)
            self.logger.record("eval/collision_rate", collision_rate)
            self.logger.record("eval/offroad_rate", offroad_rate)
            self.logger.record("eval/survival_rate", survival_rate)
            self.logger.dump(self.num_timesteps)
        
        return mean_r, survival_rate, success_rate
    
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            mean_reward, survival_rate, success_rate = self.evaluate_policy_verbose()
            self.last_mean_reward = mean_reward
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                print(f"  New best mean reward: {mean_reward:.2f} — saving model")
                if self.best_model_save_path is not None:
                    self.model.save(
                        os.path.join(self.best_model_save_path, "best_model")
                    )
            
            if mean_reward != self._prev_mean_reward:
                self._prev_mean_reward = mean_reward
                all_met = self.curriculum_cb.on_eval_end(mean_reward, survival_rate, success_rate)
            
            if self.early_stopping_cb is not None:
                stage = self.curriculum_cb.curriculum[self.curriculum_cb.current_stage_idx]
                all_met = (
                    mean_reward >= stage.reward_threshold and
                    survival_rate >= stage.min_survival_rate and
                    success_rate >= stage.min_success_rate
                )
                if not self.early_stopping_cb.on_eval_end(all_met):
                    return False
        
        return True


class EarlyStoppingCallback(BaseCallback):
    """
    Stops training once the final curriculum stage thresholds have been
    met for `patience` consecutive evaluations.
    """
    def __init__(
        self,
        curriculum_cb: CurriculumCallback,
        patience: int = 2,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.curriculum_cb = curriculum_cb
        self.patience = patience
        self._consecutive_passes = 0
    
    def _on_step(self) -> bool:
        return True
    
    def on_eval_end(self, all_thresholds_met: bool) -> bool:
        """
        Called by CurriculumEvalCallback after each evaluation.
        Returns False (stop) once patience is exhausted on final stage.
        """
        on_final_stage = (
            self.curriculum_cb.current_stage_idx
            == len(self.curriculum_cb.curriculum) - 1
        )
        
        if on_final_stage and all_thresholds_met:
            self._consecutive_passes += 1
            if self.verbose:
                print(
                    f"[EarlyStopping] Final stage passed "
                    f"({self._consecutive_passes}/{self.patience})"
                )
            if self._consecutive_passes >= self.patience:
                print(
                    f"[EarlyStopping] Threshold met {self.patience} consecutive "
                    f"time(s) on final stage — stopping training early."
                )
                return False
        else:
            self._consecutive_passes = 0
        
        return True
