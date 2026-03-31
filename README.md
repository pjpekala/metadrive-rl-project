# MetaDrive RL Project

This project aims to develop an autonomous driving agent by applying deep reinforcement learning techniques within the [MetaDrive](https://metadriverse.github.io/metadrive/) simulation environment.

_This project installs the MetaDrive simulator directly from its GitHub repository to ensure the latest compatible version is used._

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) - package manager
- [just](https://github.com/casey/just) - command runner

#### Installing just

```bash
# macOS
brew install just

# Windows (using winget)
winget install just.just

# Or via cargo/rust
cargo install just

# Or download a release binary from https://github.com/casey/just/releases
```

### Installation

1.  **Create and activate a virtual environment:**

    ```bash
    uv venv
    source .venv/bin/activate
    ```

    _(On Windows, use `.venv\Scripts\activate`)_

2.  **Install dependencies:**
    ```bash
    uv sync
    ```

## Usage

### Quick Start

```bash
# Train discrete (simpler action space)
just train discrete

# Train continuous (full action space, uses discrete trunk)
just train continuous

# Evaluate trained model
just eval continuous
```

### Training

| Command | Description |
|---------|-------------|
| `just train discrete` | Train with discrete actions (steering/throttle discretized) |
| `just train continuous` | Train with continuous actions (transfers trunk from discrete) |
| `just train all` | Train discrete then continuous sequentially |

Training saves to:
- Best model: `models/best_{phase}/best_model.zip`
- Checkpoints: `models/checkpoints_{phase}/`
- Final model: `models/{phase}/final_model.zip`

### Evaluation

| Command | Description |
|---------|-------------|
| `just eval continuous` | Evaluate on final stage (default) |
| `just eval continuous --stage 2` | Evaluate on specific stage (1-based, see below) |
| `just eval continuous --all-stages` | Sweep all stages |
| `just eval continuous --render` | Watch the agent drive |

**Stage indexing is 1-based** (e.g., `--stage 1` is the first stage, `--stage 2` is the second). The number of available stages depends on the curriculum.

### SAC (Soft Actor-Critic) Training

The project includes a modern SAC training framework with support for multiple sensor types. Located in `src/sac_training/`, this provides an alternative to PPO with built-in multi-sensor support.

#### Quick Start

```bash
# Train with BEV (Bird's Eye View) - semantic maps with CNN
uv run src/sac_training/train.py --sensor-type bev

# Train with LiDAR (default, fastest baseline)
uv run src/sac_training/train.py --sensor-type lidar

# Train with RGB camera
uv run src/sac_training/train.py --sensor-type rgb

# Train with Dense-LiDAR (high-resolution point cloud)
uv run src/sac_training/train.py --sensor-type dense-lidar
```

#### Supported Sensors

| Sensor | Observation | Policy | Best For | Speed |
|--------|-------------|--------|----------|-------|
| **LiDAR** | 240-point cloud vector | MlpPolicy | Fast baseline, interpretability | ⚡⚡⚡ Fast |
| **BEV** ✨ | 84×84×5 semantic map | CnnPolicy | Spatial reasoning, semantic info | ⚡⚡ Medium |
| **RGB** | 84×84×3 camera image | CnnPolicy | Human vision similarity, sim-to-real | ⚡⚡ Medium |
| **Dense-LiDAR** | 720-point cloud vector | MlpPolicy | Fine-grained obstacle detection | ⚡ Slowest |

**BEV (Bird's Eye View)** is the new recommended sensor type. It provides 5-channel semantic maps:
- Channel 0: Road surface
- Channel 1: Lane markings
- Channel 2: Navigation route
- Channel 3: Other vehicles
- Channel 4: Pedestrians

These labeled features enable the agent to learn more robust driving policies compared to raw point clouds.

#### Training Commands

```bash
# Basic training (8 parallel envs, 1M timesteps)
uv run src/sac_training/train.py --sensor-type bev

# Custom configuration
uv run src/sac_training/train.py \
  --sensor-type bev \
  --n-envs 8 \
  --total-timesteps 2_000_000 \
  --learning-rate 3e-4 \
  --batch-size 256

# With video recording (saves episodes every 50 steps)
uv run src/sac_training/train.py \
  --sensor-type bev \
  --record-video \
  --video-interval 50

# Live training (watch agent learn, requires n_envs=1)
uv run src/sac_training/train.py \
  --sensor-type lidar \
  --live-render \
  --n-envs 1

# With visualization and live plotting
uv run src/sac_training/train.py \
  --sensor-type bev \
  --visualize \
  --plot-interval 200
```

#### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sensor-type` | `lidar` | Sensor type: `lidar`, `bev`, `rgb`, or `dense-lidar` |
| `--total-timesteps` | 1,000,000 | Total environment steps (~1M = 20-30 hours on 8 envs) |
| `--n-envs` | 8 | Parallel environments (more = faster collection, more memory) |
| `--learning-rate` | 3e-4 | Actor/critic learning rate |
| `--batch-size` | 256 | Samples per gradient update (image sensors capped at 128) |
| `--buffer-size` | 1,000,000 | Replay buffer capacity (image sensors capped at 50K) |
| `--learning-starts` | 1000 | Steps before training starts (buffer population) |
| `--tau` | 0.005 | Soft update rate for target networks |
| `--gamma` | 0.99 | Discount factor for future rewards |
| `--record-video` | False | Record episode videos |
| `--video-interval` | 50 | Record video every N episodes |
| `--live-render` | False | Show live rendering during training |
| `--visualize` | False | Enable live metric visualization |
| `--plot-interval` | 200 | Update plots every N steps |

#### Evaluation

```bash
# Test a trained model
uv run src/sac_training/train.py \
  --test models/sac_metadrive_lidar/sac_metadrive_final \
  --sensor-type lidar

# Programmatic evaluation
from src.sac_training import test_trained_agent

results = test_trained_agent(
    model_path="models/sac_metadrive_bev/sac_metadrive_final",
    sensor_type="bev",
    num_episodes=50,
    render=True
)
# results: success_rate, mean_reward, mean_length, etc.
```

#### Model Checkpoints & Saving

Models are saved to `models/sac_metadrive_{sensor_type}/`:
```
models/sac_metadrive_bev/
├── sac_metadrive_120000_steps.zip     # Checkpoint at 120K steps
├── sac_metadrive_240000_steps.zip     # Checkpoint at 240K steps
├── ...
└── sac_metadrive_final.zip            # Final trained model
```

Checkpoints are saved every 120,000 steps by default. Use `--save-freq` to customize.

#### Important Notes

- **Image Sensors (BEV/RGB):** Buffer size automatically capped at 50K and batch size at 128 to prevent excessive memory usage (each image is ~35KB)
- **Live Rendering:** Only works with `--n-envs 1` (rendering multiple parallel environments isn't supported)
- **BEV Frame Stacking:** The BEV sensor stacks 3 consecutive observations for temporal information
- **Video Recording:** Spawns a separate environment; training continues in parallel
- **Sensor Selection:**
  - Use **LiDAR** for fast baselines and memory-constrained settings
  - Use **BEV** for semantic reasoning and labeled observations
  - Use **RGB** for sim-to-real transfer research
  - Use **Dense-LiDAR** for fine-grained obstacle detection

#### API Reference

```python
from src.sac_training import (
    # Configurations
    BEVConfig, LiDARConfig, RGBConfig, TrainingConfig,
    
    # Utilities
    SensorFactory, EnvironmentFactory,
    get_policy_type, is_image_observation,
    
    # Training
    SACTrainer, test_trained_agent,
    
    # Feature extractors
    BEVCNN, RGBCNN,
)

# Create a trainer
trainer = SACTrainer(
    sensor_type="bev",
    config=TrainingConfig(total_timesteps=1_000_000, n_envs=8)
)

# Train the model
model = trainer.train(visualize=True, record_video=True)
```

### Monitoring

```bash
just tensorboard  # Opens TensorBoard at localhost:6006
```

### Curriculum

The training automatically progresses through stages:

**Discrete:**
- Stage 1: Straight roads, no traffic
- Stage 2: Curves added, no traffic

**Continuous:**
- Stage 1: Straight roads, no traffic (warm-up)
- Stage 2: Curves, no traffic
- Stage 3: Mixed roads, light traffic (0.05 density)
- Stage 4: Mixed roads, moderate traffic (0.1 density)
- Stage 5: Random maps, moderate traffic (0.2 density)
- Stage 6: Random maps, heavy traffic (0.5 density)

Agent advances when success_rate and survival_rate thresholds are met.

### Model Loading & Transfer Learning

#### Weight Fallback Pattern
When loading models, the system uses a fallback priority:
1. `final_model.zip` - Model saved at the end of training
2. `best_model.zip` - Best checkpoint by evaluation reward
3. Error if neither exists

#### Discrete → Continuous Transfer
When training continuous actions, the MLP trunk weights are transferred from the discrete model:
- Only the shared feature extraction layers are transferred
- The output head (policy/value networks) is initialized fresh for continuous actions
- Fallback order: final_discrete → best_discrete → error

This transfer learning approach leverages features learned in the simpler discrete action space to speed up continuous training.

### Configuration

Edit `src/curriculum_learning/config.py` to customize:
- Stage definitions (maps, scenarios, traffic density)
- Advancement thresholds (min_success_rate, min_survival_rate)
- Training hyperparameters (learning rate, batch size, etc.)
- Model architecture (network layers)

## 1. Problem Statement

Modern autonomous driving systems face a significant challenge in generalizing to the long tail of rare and complex driving scenarios (e.g., unprotected turns, dense traffic, unusual road layouts). Training models purely on real-world data is data-intensive, expensive, and potentially unsafe. Simulation offers a scalable and safe alternative, but requires agents to learn robust policies that can handle a wide variety of situations they have not explicitly been trained on. This project addresses the problem of training a sample-efficient reinforcement learning agent that can achieve safe and effective navigation across a diverse set of procedurally generated driving environments.

## 2. Project Objectives

1.  **Primary Objective:** To successfully train a Deep Reinforcement Learning (DRL) agent that can navigate from a starting point to a destination in a variety of procedurally generated urban environments in MetaDrive without collisions.
2.  **Secondary Objective:** Achieve a task success rate of over 80% on a held-out set of unseen evaluation environments to demonstrate generalization.
3.  **Tertiary Objective:** Compare the sample efficiency and final performance of the Proximal Policy Optimization (PPO) algorithm against a potential alternative like Soft Actor-Critic (SAC).

## 3. Proposed Approach

- **Environment:** We will use the `MetaDrive-v0` environment, which provides a rich set of procedurally generated road networks and traffic conditions. The observation space will consist of a simplified bird's-eye view LiDAR-like representation provided by the environment. The action space is continuous, controlling the vehicle's steering and acceleration/braking.
- **Reward Function:** The agent will be rewarded for making progress towards the destination, with a large terminal reward for success. Penalties will be applied for collisions, driving off-road, and potentially for time, to encourage efficient routes.
- **Algorithm:** We will start by implementing the **Proximal Policy Optimization (PPO)** algorithm, using the implementation from the `stable-baselines3` library. PPO is chosen for its stability, reliability, and strong performance on a wide range of continuous control tasks.
- **Training:** The agent will be trained for a fixed number of timesteps on a set of training seeds for procedural generation. The training progress will be monitored using TensorBoard.

## 4. Experimental Design

- **Dataset:** The "dataset" will be a set of 100 procedurally generated training maps within MetaDrive.
- **Evaluation:** The trained agent will be evaluated on 20 unseen, held-out maps. The agent's performance will be measured without any learning or exploration noise.
- **Metrics:**
  - **Success Rate:** The primary metric. Percentage of episodes where the agent reaches the destination.
  - **Average Episode Reward:** To track learning progress.
  - **Collision Rate:** Percentage of episodes ending in a collision.
  - **Average Episode Length:** To measure the efficiency of the agent's policy.
- **Baseline:** The performance of the trained PPO agent will be compared against a baseline agent that takes random actions in the environment. This ensures that the learned policy is significantly better than chance.
