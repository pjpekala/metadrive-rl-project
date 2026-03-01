import os
from stable_baselines3 import PPO
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Configuration
MODEL_NAME = "ppo_metadrive"
LOG_PATH = "logs"
MODEL_SAVE_PATH = f"models/{MODEL_NAME}"
TOTAL_TIMESTEPS = 100000  # Lower for a quick test run

if __name__ == "__main__":
    env = None  # Initialize env to None
    try:
        # 1. Create the MetaDrive environment
        # The 'use_render=True' is commented out during training for speed.
        # Enable it in evaluate.py to see the agent drive.
        env = SafeMetaDriveEnv(config={"num_scenarios": 1, "start_seed": 0, "use_render": False})

        # 2. Create the PPO agent
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=LOG_PATH,
        )

        # 3. Train the agent
        print("--- Starting Training ---")
        model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
        print("--- Training Finished ---")

        # 4. Save the trained model
        model.save(MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}.zip")
    finally:
        if env is not None:
            env.close()
