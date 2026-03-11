"""
curriculum_lib.py

Shared environment, callback, and training logic used by train.py.
Nothing in here needs to be edited between runs — all configuration
lives in train.py.

Environment alignment guarantee:
  Training env, inline eval env, and evaluate.py all use the same
  _build_config() helper so reward shaping, termination flags, and
  action space are always identical for a given stage.
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv


# ---------------------------------------------------------------------------
# Shared config builder — single source of truth for all three env types
# ---------------------------------------------------------------------------

def build_metadrive_config(stage, seed, scenario_index=0, use_render=False):
    """
    Builds a MetaDrive config dict for a given curriculum stage.
    Used by CurriculumEnv, make_eval_env, and evaluate.py so that
    reward shaping, termination flags, and action space are always
    identical across training, inline eval, and held-out eval.

    stage          -- curriculum stage dict from train.py
    seed           -- base seed for this env worker
    scenario_index -- worker index; offsets seed so workers see different scenarios
    use_render     -- set True only in evaluate.py --render mode
    """
    num_scenarios = stage["num_scenarios"]
    config = {
        "num_scenarios":         num_scenarios,
        "start_seed":            seed + scenario_index * num_scenarios,
        "map":                   stage["map"],
        "traffic_density":       stage["traffic_density"],

        # --- Reward shaping ---
        "driving_reward":        1.0,   # dense per-step forward progress
        "speed_reward":          0.1,   # small bonus to prevent stopping
        "use_lateral_reward":    True,  # pulls car toward lane center
        "success_reward":        10.0,
        "out_of_road_penalty":   5.0,
        "crash_vehicle_penalty": 5.0,

        # --- Termination --- (must be explicit; MetaDrive defaults can differ)
        "out_of_road_done":      True,
        "crash_vehicle_done":    True,
        "crash_object_done":     True,

        "horizon":               1000,
        "use_render":            use_render,
        "log_level":             50,

        # --- Action space ---
        "discrete_action":       stage["discrete_action"],
    }
    if stage["discrete_action"]:
        config["discrete_steering_dim"] = stage["discrete_steering_dim"]
        config["discrete_throttle_dim"] = stage["discrete_throttle_dim"]
    return config


# ---------------------------------------------------------------------------
# Training environment
# ---------------------------------------------------------------------------

class CurriculumEnv(SafeMetaDriveEnv):
    """
    MetaDrive env that supports live curriculum stage updates without
    subprocess teardown. Each worker holds a reference to a shared
    `stage_holder` list. On every reset(), if the stage has changed since
    the last episode, the env re-initialises its config in-place.
    """
    def __init__(self, config, train_seed, scenario_index, stage_holder):
        self.train_seed = train_seed
        self.scenario_index = scenario_index
        self.stage_holder = stage_holder
        self._current_stage_name = None
        super().__init__(
            config=build_metadrive_config(stage_holder[0], train_seed, scenario_index)
        )

    def reset(self, **kwargs):
        # Drop SB3's seed kwarg — MetaDrive validates seed against its own
        # scenario window and raises AssertionError if it's out of range
        kwargs.pop("seed", None)
        kwargs.pop("options", None)

        stage = self.stage_holder[0]
        if stage["name"] != self._current_stage_name:
            self._current_stage_name = stage["name"]
            self.config.update(
                build_metadrive_config(stage, self.train_seed, self.scenario_index)
            )
        return super().reset(**kwargs)

    def step(self, action):
        try:
            return super().step(action)
        except Exception as e:
            print(f"[CurriculumEnv] step() error: {e} — treating as termination")
            obs = super().reset()[0]
            return obs, -10.0, True, False, {}


def make_env(train_seed, scenario_index, stage_holder):
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


# ---------------------------------------------------------------------------
# Inline eval environment
# ---------------------------------------------------------------------------

class InlineEvalEnv(SafeMetaDriveEnv):
    """
    Eval env that mirrors CurriculumEnv's stage-tracking behaviour so the
    inline EvalCallback always evaluates on the same stage the agent is
    currently training on — not the frozen Stage 0 config from startup.
    """
    def __init__(self, eval_seed, stage_holder):
        self.eval_seed = eval_seed
        self.stage_holder = stage_holder
        self._current_stage_name = None
        super().__init__(
            config=build_metadrive_config(stage_holder[0], eval_seed)
        )

    def reset(self, **kwargs):
        kwargs.pop("seed", None)
        kwargs.pop("options", None)

        stage = self.stage_holder[0]
        if stage["name"] != self._current_stage_name:
            self._current_stage_name = stage["name"]
            self.config.update(build_metadrive_config(stage, self.eval_seed))
        return super().reset(**kwargs)

    def step(self, action):
        try:
            return super().step(action)
        except Exception as e:
            print(f"[InlineEvalEnv] step() error: {e} — treating as termination")
            obs = super().reset()[0]
            return obs, -10.0, True, False, {}


def make_inline_eval_env(eval_seed, stage_holder):
    """
    Factory for the inline eval env. Tracks stage_holder so it always
    evaluates on the current curriculum stage, not the initial one.
    """
    def _init():
        env = InlineEvalEnv(eval_seed=eval_seed, stage_holder=stage_holder)
        env = Monitor(env)
        return env
    return _init


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class CurriculumCallback(BaseCallback):
    """
    Syncs VecNormalize obs stats to the eval env on the eval schedule and
    advances the curriculum stage when all thresholds are met.
    """
    def __init__(self, train_env, eval_env, eval_freq,
                 stage_holder, curriculum, vecnorm_ckpt_path, verbose=1):
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
            # Sync obs normalizer stats so eval sees the same scaling as training
            self.eval_env.obs_rms = self.train_env.obs_rms
            self.eval_env.training = False
            self.train_env.save(self.vecnorm_ckpt_path)

        # Log current stage index every step for TensorBoard timeline visibility
        self.logger.record("curriculum/stage", self.current_stage_idx)
        return True

    def on_eval_end(self, mean_reward, survival_rate, success_rate):
        self.eval_count += 1
        stage = self.curriculum[self.current_stage_idx]
        reward_ok   = mean_reward   >= stage["reward_threshold"]
        survival_ok = survival_rate >= stage["min_survival_rate"]
        success_ok  = success_rate  >= stage["min_success_rate"]

        steps_in_stage = self.num_timesteps - self.stage_start_timestep

        print(f"\n[Curriculum] {stage['name']} (eval #{self.eval_count})")
        print(f"  Steps in stage: {steps_in_stage:,}")
        print(f"  Reward:   {mean_reward:.1f} (need >= {stage['reward_threshold']}) — {'✓' if reward_ok else '✗'}")
        print(f"  Survival: {survival_rate*100:.0f}% (need >= {stage['min_survival_rate']*100:.0f}%) — {'✓' if survival_ok else '✗'}")
        print(f"  Success:  {success_rate*100:.0f}% (need >= {stage['min_success_rate']*100:.0f}%) — {'✓' if success_ok else '✗'}")

        # Log thresholds alongside actuals for TensorBoard overlay
        self.logger.record("curriculum/stage",              self.current_stage_idx)
        self.logger.record("curriculum/survival_threshold", stage["min_survival_rate"])
        self.logger.record("curriculum/success_threshold",  stage["min_success_rate"])
        self.logger.dump(self.num_timesteps)

        if (reward_ok and survival_ok and success_ok
                and self.current_stage_idx < len(self.curriculum) - 1):
            self.current_stage_idx += 1
            new_stage = self.curriculum[self.current_stage_idx]
            self.stage_start_timestep = self.num_timesteps
            print(f"\n{'='*60}")
            print(f"[Curriculum] ✓ ADVANCING to: {new_stage['name']}")
            print(f"{'='*60}\n")
            self.stage_holder[0] = new_stage


class CurriculumEvalCallback(EvalCallback):
    """
    Extends EvalCallback with per-episode logging and notifies
    CurriculumCallback once per completed evaluation run.
    """
    def __init__(self, curriculum_cb, early_stopping_cb=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum_cb      = curriculum_cb
        self.early_stopping_cb  = early_stopping_cb
        self._prev_mean_reward  = None

    def evaluate_policy_verbose(self):
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
                crash       = info[0].get("crash",       False)
                out_of_road = info[0].get("out_of_road", False)

                if arrive_dest:          success_count   += 1
                if crash:                collision_count += 1
                if out_of_road:          offroad_count   += 1
                if crash or out_of_road: failed_count    += 1

                if arrive_dest:   status = "✓ SUCCESS"
                elif crash:       status = "✗ CRASH  "
                elif out_of_road: status = "~ OFFROAD"
                else:             status = "  TIMEOUT"

                print(
                    f"  Ep {ep_num:2d} | {status} | "
                    f"Reward: {ep_reward:7.2f} | "
                    f"Length: {ep_length:4d}"
                )
                ep_reward, ep_length = 0.0, 0
                obs = self.eval_env.reset()

        n              = self.n_eval_episodes
        mean_r         = np.mean(episode_rewards)
        std_r          = np.std(episode_rewards)
        mean_len       = np.mean(episode_lengths)
        survival_rate  = (n - failed_count) / n
        success_rate   = success_count / n
        collision_rate = collision_count / n
        offroad_rate   = offroad_count / n

        print(
            f"  Summary | Mean reward: {mean_r:.2f} ± {std_r:.2f} | "
            f"Success: {success_count}/{n} | "
            f"Crash: {collision_count}/{n} | "
            f"Off-road: {offroad_count}/{n} | "
            f"Survival: {int(survival_rate*n)}/{n}\n"
        )

        if self.logger is not None:
            self.logger.record("eval/mean_reward",    mean_r)
            self.logger.record("eval/std_reward",     std_r)
            self.logger.record("eval/mean_ep_length", mean_len)
            self.logger.record("eval/success_rate",   success_rate)
            self.logger.record("eval/collision_rate", collision_rate)
            self.logger.record("eval/offroad_rate",   offroad_rate)
            self.logger.record("eval/survival_rate",  survival_rate)
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
                self.curriculum_cb.on_eval_end(mean_reward, survival_rate, success_rate)

            # Notify early stopping — check whether all thresholds were met
            if self.early_stopping_cb is not None:
                stage = self.curriculum_cb.curriculum[self.curriculum_cb.current_stage_idx]
                all_met = (
                    mean_reward   >= stage["reward_threshold"] and
                    survival_rate >= stage["min_survival_rate"] and
                    success_rate  >= stage["min_success_rate"]
                )
                if not self.early_stopping_cb.on_eval_end(all_met):
                    return False  # propagate stop signal to model.learn()

        return True


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStoppingCallback(BaseCallback):
    """
    Stops training once the final curriculum stage thresholds have been
    met for `patience` consecutive evaluations. Prevents wasting compute
    after the curriculum has fully converged.

    Wired into CurriculumEvalCallback.on_eval_passed() — do not call
    _on_step() directly for stopping logic; the signal flows through
    CurriculumEvalCallback returning False from _on_step().
    """
    def __init__(self, curriculum_cb, patience=2, verbose=1):
        super().__init__(verbose)
        self.curriculum_cb      = curriculum_cb
        self.patience           = patience
        self._consecutive_passes = 0

    def _on_step(self) -> bool:
        return True

    def on_eval_end(self, all_thresholds_met) -> bool:
        """
        Called by CurriculumEvalCallback after each evaluation round.
        Returns False (stop signal) once patience is exhausted on the
        final stage; returns True (continue) otherwise.
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
                return False  # caller should propagate as stop signal
        else:
            # Reset counter if we advance stage or fail thresholds
            self._consecutive_passes = 0

        return True


# ---------------------------------------------------------------------------
# Trunk transfer
# ---------------------------------------------------------------------------

def transfer_trunk(source_ckpt, source_vecnorm, target_env,
                   stage_holder, train_seed, ppo_kwargs):
    """
    Loads a checkpoint, copies the MLP trunk weights and obs normalizer
    stats into a fresh PPO model built against target_env, then returns it.
    The output head is left freshly initialized (needed when switching
    from discrete to continuous action space).
    """
    print(f"[transfer_trunk] Loading source checkpoint: {source_ckpt}")

    tmp_vec = SubprocVecEnv([make_env(train_seed, 0, stage_holder)])
    tmp_env = VecNormalize.load(source_vecnorm, tmp_vec)
    source_model = PPO.load(source_ckpt, env=tmp_env, device=ppo_kwargs["device"])
    trunk_state  = source_model.policy.mlp_extractor.state_dict()
    obs_rms      = tmp_env.obs_rms
    tmp_env.close()

    model = PPO("MlpPolicy", target_env, **ppo_kwargs)
    model.policy.mlp_extractor.load_state_dict(trunk_state)
    target_env.obs_rms = obs_rms

    print("[transfer_trunk] Trunk weights transferred. Output head initialized fresh.")
    return model


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def run_training(
    curriculum,
    run_name,
    n_envs,
    n_steps,
    batch_size,
    total_timesteps,
    train_seed,
    eval_seed,
    device,
    eval_freq_steps=100_000,
    n_eval_episodes=20,
    trunk_source_ckpt=None,
    trunk_source_vecnorm=None,
    ppo_overrides=None,
):
    # --- Derived paths ---
    model_dir         = f"models/{run_name}"
    log_path          = f"logs/run_{run_name}"
    best_dir          = f"models/best_{run_name}"
    ckpt_dir          = f"models/checkpoints_{run_name}"
    checkpoint_path   = f"{best_dir}/best_model.zip"
    vecnorm_ckpt_path = f"{best_dir}/vecnorm.pkl"
    final_model_path  = f"{model_dir}/final_model"
    final_vecnorm_path = f"{model_dir}/final_vecnorm.pkl"

    for d in [model_dir, log_path, best_dir, ckpt_dir, "logs"]:
        os.makedirs(d, exist_ok=True)

    eval_freq = eval_freq_steps // n_envs

    # --- Default PPO hyperparameters ---
    ppo_kwargs = dict(
        n_steps=n_steps, batch_size=batch_size, n_epochs=10,
        learning_rate=1e-4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.1, ent_coef=0.01, vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02,   # caps update size; prevents the KL/clip blowup seen in run 1
        device=device, verbose=1,
        tensorboard_log=log_path,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )
    if ppo_overrides:
        ppo_kwargs.update(ppo_overrides)

    # --- Build vectorized envs ---
    stage_holder = [curriculum[0]]
    print(f"\n[{run_name}] Starting at: {stage_holder[0]['name']}")
    print(f"[{run_name}] Device: {device} | Envs: {n_envs} | Steps: {total_timesteps:,}")

    train_vec = SubprocVecEnv([
        make_env(train_seed, i, stage_holder) for i in range(n_envs)
    ])

    # Inline eval uses InlineEvalEnv which tracks stage_holder, so it always
    # evaluates on the current stage rather than the frozen startup stage
    eval_vec = SubprocVecEnv([make_inline_eval_env(eval_seed, stage_holder)])

    # --- Model initialisation: three-way priority ---
    #   1. Resume from a local best checkpoint (interrupted run)
    #   2. Transfer trunk from a source checkpoint (discrete → continuous)
    #   3. Train from scratch
    if os.path.exists(checkpoint_path) and os.path.exists(vecnorm_ckpt_path):
        print(f"[{run_name}] Resuming from {checkpoint_path}")
        train_env = VecNormalize.load(vecnorm_ckpt_path, train_vec)
        train_env.training = True
        train_env.norm_reward = True
        model = PPO.load(
            checkpoint_path, env=train_env, device=device,
            custom_objects={
                "learning_rate": ppo_kwargs["learning_rate"],
                "clip_range":    ppo_kwargs["clip_range"],
                "n_steps":       n_steps,
                "batch_size":    batch_size,
            }
        )

    elif trunk_source_ckpt and os.path.exists(trunk_source_ckpt):
        print(f"[{run_name}] Transferring trunk from {trunk_source_ckpt}")
        train_env = VecNormalize(
            train_vec, norm_obs=True, norm_reward=True,
            clip_obs=10.0, clip_reward=10.0,
        )
        model = transfer_trunk(
            source_ckpt=trunk_source_ckpt,
            source_vecnorm=trunk_source_vecnorm,
            target_env=train_env,
            stage_holder=stage_holder,
            train_seed=train_seed,
            ppo_kwargs=ppo_kwargs,
        )

    else:
        print(f"[{run_name}] No checkpoint found — training from scratch")
        train_env = VecNormalize(
            train_vec, norm_obs=True, norm_reward=True,
            clip_obs=10.0, clip_reward=10.0,
        )
        model = PPO("MlpPolicy", train_env, **ppo_kwargs)

    # clip_obs matches training so normalized obs are on the same scale
    eval_env = VecNormalize(
        eval_vec, norm_obs=True, norm_reward=False,
        clip_obs=10.0, training=False,
    )
    eval_env.obs_rms = train_env.obs_rms

    # --- Callbacks ---
    curriculum_cb = CurriculumCallback(
        train_env=train_env, eval_env=eval_env,
        eval_freq=eval_freq, stage_holder=stage_holder,
        curriculum=curriculum, vecnorm_ckpt_path=vecnorm_ckpt_path,
        verbose=1,
    )
    early_stopping_cb = EarlyStoppingCallback(
        curriculum_cb=curriculum_cb,
        patience=2,   # final stage must pass thresholds 2 evals in a row to stop
        verbose=1,
    )
    eval_cb = CurriculumEvalCallback(
        curriculum_cb=curriculum_cb,
        early_stopping_cb=early_stopping_cb,
        eval_env=eval_env,
        best_model_save_path=best_dir,
        log_path=log_path,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=eval_freq,
        save_path=ckpt_dir,
        name_prefix=f"ppo_{run_name}",
    )

    # --- Train ---
    model.learn(
        total_timesteps=total_timesteps,
        callback=[curriculum_cb, eval_cb, checkpoint_cb, early_stopping_cb],
        progress_bar=True,
        reset_num_timesteps=False,
    )
    print(f"\n[{run_name}] Training finished.")

    model.save(final_model_path)
    train_env.save(final_vecnorm_path)
    print(f"[{run_name}] Model:   {final_model_path}.zip")
    print(f"[{run_name}] VecNorm: {final_vecnorm_path}")

    train_env.close()
    eval_env.close()