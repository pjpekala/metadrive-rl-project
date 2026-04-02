import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("models/sparse_lidar/training_log.csv")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("SAC Sparse LiDAR Training Progress (1M Timesteps)", fontsize=14, fontweight="bold")

axes[0, 0].plot(df["timestep"] / 1e6, df["mean_reward"], color="#2ecc71", linewidth=1.5)
axes[0, 0].set_xlabel("Timesteps (M)")
axes[0, 0].set_ylabel("Mean Reward")
axes[0, 0].set_title("Episode Reward")
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(df["timestep"] / 1e6, df["success_rate"], color="#3498db", linewidth=1.5, label="Success")
axes[0, 1].plot(df["timestep"] / 1e6, df["collision_rate"], color="#e74c3c", linewidth=1.5, label="Collision")
axes[0, 1].plot(df["timestep"] / 1e6, df["out_of_road_rate"], color="#f39c12", linewidth=1.5, label="Out of Road")
axes[0, 1].set_xlabel("Timesteps (M)")
axes[0, 1].set_ylabel("Rate")
axes[0, 1].set_title("Episode Outcomes")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(df["timestep"] / 1e6, df["mean_speed"], color="#9b59b6", linewidth=1.5)
axes[1, 0].set_xlabel("Timesteps (M)")
axes[1, 0].set_ylabel("Mean Speed (m/s)")
axes[1, 0].set_title("Mean Speed")
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(df["timestep"] / 1e6, df["mean_episode_length"], color="#1abc9c", linewidth=1.5)
axes[1, 1].set_xlabel("Timesteps (M)")
axes[1, 1].set_ylabel("Steps")
axes[1, 1].set_title("Episode Length")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("SAC_SparceLidar_1M.jpg", dpi=150, bbox_inches="tight")
print("Saved: SAC_SparceLidar_1M.jpg")
