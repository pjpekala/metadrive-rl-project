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

# Train continuous (full action space, uses discrete as trunk)
just train continuous

# Evaluate trained model
just eval continuous
```

### Training

| Command | Description |
|---------|-------------|
| `just train discrete` | Train with discrete actions (steering/throttle discretized) |
| `just train continuous` | Train with continuous actions (uses discrete as trunk) |
| `just train all` | Train discrete then continuous sequentially |

Training saves to:
- Best model: `models/best_{phase}/best_model.zip`
- Checkpoints: `models/checkpoints_{phase}/`
- Final model: `models/{phase}/final_model.zip`

### Evaluation

| Command | Description |
|---------|-------------|
| `just eval continuous` | Evaluate on final stage (default) |
| `just eval continuous --stage 2` | Evaluate on specific stage |
| `just eval continuous --all-stages` | Sweep all stages |
| `just eval continuous --render` | Watch the agent drive |

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
