from stable_baselines3 import PPO
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv

# Configuration
MODEL_NAME = "ppo_metadrive"
MODEL_LOAD_PATH = f"models/{MODEL_NAME}.zip"

if __name__ == "__main__":
    env = None # Initialize env to None
    try:
        # 1. Create the MetaDrive environment for evaluation
        # use_render=True allows us to see the agent's driving.
        # We use a different set of maps (start_seed=1000) to test generalization.
        env = SafeMetaDriveEnv(config={"num_scenarios": 20, "start_seed": 1000, "use_render": True})

        # 2. Load the trained PPO agent
        try:
            model = PPO.load(MODEL_LOAD_PATH, env=env)
            print(f"Model loaded from {MODEL_LOAD_PATH}")
        except FileNotFoundError:
            print(f"Error: Model not found at {MODEL_LOAD_PATH}")
            print("Please run the training script (src/train.py) first.")
            exit()


        # 3. Evaluate the agent
        print("--- Starting Evaluation ---")
        obs, info = env.reset()
        terminated = False
        truncated = False
        ep_reward = 0
        
        for i in range(5000): # Max steps per episode
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            if terminated or truncated:
                print(f"Episode finished. Reward: {ep_reward}, Info: {info}")
                obs, info = env.reset()
                ep_reward = 0
        
        print("--- Evaluation Finished ---")
    finally:
        if env is not None:
            env.close()
