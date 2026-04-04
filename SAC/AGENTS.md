# MetaDrive RL Training Project

## Environment Setup
This project uses a conda environment located at:
`/home/kaivalyajadhav/miniconda3/envs/envmetadrive`

\`\`\`bash
# Activate conda environment
conda activate envmetadrive

# Or use the full path to Python
/home/kaivalyajadhav/miniconda3/envs/envmetadrive/bin/python
\`\`\`

To recreate the environment from scratch, save the following as \`environment.yml\` and run \`conda env create -f environment.yml\`:

\`\`\`yaml
name: envmetadrive
channels:
  - defaults
dependencies:
  - _libgcc_mutex=0.1=main
  - _openmp_mutex=5.1=51_gnu
  - bzip2=1.0.8=h998d150_6
  - ca-certificates=2025.12.2=hd43f75c_0
  - ld_impl_linux-aarch64=2.44=h97084ae_3
  - libexpat=2.7.5=h5b11e9f_0
  - libffi=3.4.4=h419075a_1
  - libgcc=15.2.0=hc18542e_7
  - libgcc-ng=15.2.0=h51576c1_7
  - libgomp=15.2.0=hf47c802_7
  - libstdcxx=15.2.0=h9538471_7
  - libstdcxx-ng=15.2.0=h719ae8e_7
  - libuuid=1.41.5=h998d150_0
  - libxcb=1.17.0=hf66535e_0
  - libzlib=1.3.1=h998d150_0
  - ncurses=6.5=h419075a_0
  - openssl=3.5.5=h39c9139_0
  - packaging=25.0=py312hd43f75c_1
  - pip=26.0.1=pyhc872135_0
  - pthread-stubs=0.3=hfd63f10_1
  - python=3.12.13=h51a4668_1
  - readline=8.3=h886d1d0_0
  - setuptools=80.10.2=py312hd43f75c_0
  - sqlite=3.51.2=h6a2ceb9_0
  - tk=8.6.15=h987d8db_0
  - tzdata=2026a=he532380_0
  - wheel=0.46.3=py312hd43f75c_0
  - xorg-libx11=1.8.12=hf66535e_1
  - xorg-libxau=1.0.12=hf66535e_0
  - xorg-libxdmcp=1.1.5=hf66535e_0
  - xorg-xorgproto=2024.1=h998d150_1
  - xz=5.8.2=hed4fdd2_0
  - zlib=1.3.1=h998d150_0
  - pip:
    - absl-py==2.4.0
    - ale-py==0.11.2
    - asttokens==3.0.1
    - certifi==2026.2.25
    - charset-normalizer==3.4.6
    - cloudpickle==3.1.2
    - contourpy==1.3.3
    - cuda-bindings==13.2.0
    - cuda-pathfinder==1.5.0
    - cuda-toolkit==13.0.2
    - cycler==0.12.1
    - decorator==5.2.1
    - executing==2.2.1
    - farama-notifications==0.0.4
    - filelock==3.25.2
    - fonttools==4.62.1
    - fsspec==2026.2.0
    - grpcio==1.78.0
    - gymnasium==1.2.3
    - idna==3.11
    - imageio==2.37.3
    - imageio-ffmpeg==0.6.0
    - ipython==9.11.0
    - ipython-pygments-lexers==1.1.1
    - jedi==0.19.2
    - jinja2==3.1.6
    - kiwisolver==1.5.0
    - lxml==6.0.2
    - markdown==3.10.2
    - markdown-it-py==4.0.0
    - markupsafe==3.0.3
    - matplotlib==3.10.8
    - matplotlib-inline==0.2.1
    - mdurl==0.1.2
    - mediapy==1.2.6
    - metadrive-simulator==0.4.3
    - mpmath==1.3.0
    - networkx==3.6.1
    - numpy==2.4.3
    - nvidia-cublas==13.1.0.3
    - nvidia-cuda-cupti==13.0.85
    - nvidia-cuda-nvrtc==13.0.88
    - nvidia-cuda-runtime==13.0.96
    - nvidia-cudnn-cu13==9.19.0.56
    - nvidia-cufft==12.0.0.61
    - nvidia-cufile==1.15.1.6
    - nvidia-curand==10.4.0.35
    - nvidia-cusolver==12.0.4.66
    - nvidia-cusparse==12.6.3.3
    - nvidia-cusparselt-cu13==0.8.0
    - nvidia-nccl-cu13==2.28.9
    - nvidia-nvjitlink==13.0.88
    - nvidia-nvshmem-cu13==3.4.5
    - nvidia-nvtx==13.0.85
    - opencv-python==4.13.0.92
    - panda3d==1.10.16
    - panda3d-gltf==0.15.0
    - panda3d-simplepbr==0.13.1
    - pandas==3.0.1
    - parso==0.8.6
    - pexpect==4.9.0
    - pillow==12.1.1
    - platformdirs==4.9.4
    - progressbar==2.5
    - prompt-toolkit==3.0.52
    - protobuf==7.34.1
    - psutil==7.2.2
    - ptyprocess==0.7.0
    - pure-eval==0.2.3
    - pygame==2.6.1
    - pygments==2.19.2
    - pyparsing==3.3.2
    - python-dateutil==2.9.0.post0
    - requests==2.33.0
    - rich==14.3.3
    - shapely==2.1.2
    - six==1.17.0
    - stable-baselines3==2.7.1
    - stack-data==0.6.3
    - sympy==1.14.0
    - tensorboard==2.20.0
    - tensorboard-data-server==0.7.2
    - torch==2.11.0
    - tqdm==4.67.3
    - traitlets==5.14.3
    - triton==3.6.0
    - typing-extensions==4.15.0
    - urllib3==2.6.3
    - wcwidth==0.6.0
    - werkzeug==3.1.7
    - yapf==0.43.0
\`\`\`

---

## Project Overview
This project trains SAC (Soft Actor-Critic) agents for autonomous driving in the MetaDrive simulator.

### Project Files
| File | Purpose |
|------|---------|
| `train_metadrive.py` | Main training script with SAC algorithm |
| `evaluate_metadrive.py` | Evaluation script with video recording |

### Key Features
- **Multi-modal Sensors**: Sparse LiDAR (80 beams), Dense LiDAR (240 beams), BEV (128x128), RGB Camera
- **Auto-resume**: Loads existing model + replay buffer on restart
- **Cumulative Logging**: CSV logs track total timesteps across training sessions
- **Interrupt Safety**: Saves model + replay buffer on Ctrl+C
- **Checkpointing**: Periodic checkpoint saves during training
- **Seed Split**: Training seeds 0-799 (800 unique maps), evaluation seeds 800-999 (200 unique maps)
- **Evaluation**: Separate evaluation script (\`evaluate_metadrive.py\`) for assessing trained models

---

## Usage
### Training
\`\`\`bash
# Train Sparse LiDAR (default)
python train_metadrive.py --modality sparse_lidar --total-timesteps 1000000

# Train Dense LiDAR
python train_metadrive.py --modality dense_lidar --total-timesteps 1000000

# Train BEV (CNN)
python train_metadrive.py --modality bev --total-timesteps 1000000

# Train RGB Camera (CNN)
python train_metadrive.py --modality rgb --total-timesteps 1000000

# Resume interrupted training (automatically loads model + replay buffer)
python train_metadrive.py --modality sparse_lidar
\`\`\`

### Evaluation
\`\`\`bash
# Evaluate with video recording
python evaluate_metadrive.py --model-path ./models/sparse_lidar/sac_metadrive_final.zip --modality sparse_lidar --save-video

# Evaluate with rendering
python evaluate_metadrive.py --model-path ./models/sparse_lidar/sac_metadrive_final.zip --modality sparse_lidar --render

# Evaluate with specific number of episodes
python evaluate_metadrive.py --model-path ./models/sparse_lidar/sac_metadrive_final.zip --modality sparse_lidar --episodes 100
\`\`\`

---

## Modality Configs

| Modality | Beams/Size | Distance | Policy |
|----------|-----------|----------|--------|
| `sparse_lidar` | 80 beams | 50m | MlpPolicy |
| `dense_lidar` | 240 beams | 100m | MlpPolicy |
| `bev` | 128x128 | - | CnnPolicy |
| `rgb` | 128x128 | - | CnnPolicy |

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_timesteps` | 1,000,000 | Total training steps |
| `learning_rate` | 3e-4 | Adam learning rate |
| `buffer_size` | 1,000,000 | Replay buffer size |
| `batch_size` | 256 | Training batch size |
| `n_envs` | 8 | Parallel environments |

---

## Output Structure

\`\`\`
models/
├── sparse_lidar/
│   ├── sac_metadrive_final.zip   # Final trained model
│   ├── replay_buffer.pkl         # Replay buffer for resume
│   ├── training_log.csv          # Training metrics (appends on resume)
│   ├── checkpoints/              # Periodic checkpoints
│   └── logs/                     # TensorBoard logs
├── dense_lidar/
├── bev/
└── rgb/
\`\`\`

---

## Development Workflow (For Agents)

### 1. Development Commands
For all development operations, use the dedicated \`justfile\` in *this* directory:
- **Setup Dependencies**: \`$ just setup\` (Synchronizes environment using \`uv\`).
- **Linting**: \`$ just lint\` (Checks code style using \`ruff\`).
- **Testing**: \`$ just test\` (Runs the full test suite using \`pytest\`.\`).
- **Single Test**: \`$ just test-single -- <path/to/test_file.py>\` (For focused debugging).
- **Cleanup**: \`$ just clean\` (Removes generated artifacts like checkpoints and logs).

### 2. Code Style Guidelines
*   **Imports**: Use absolute imports where possible. Group third-party, local, and standard library imports, separating them with a blank line.
*   **Formatting**: Adhere to PEP 8 standards. Use the \`just lint\` command to enforce formatting.
*   **Type Hinting**: All function signatures, class attributes, and function returns *must* have type hints.
*   **Naming Conventions**:
    *   Modules/Files: \`snake_case\`.
    *   Classes: \`PascalCase\`.
    *   Functions/Variables: \`snake_case\`.
*   **Error Handling**: Use \`try...except\` blocks with specific exception catching (e.g., \`except FileNotFoundError\`), rather than a blanket \`except Exception:\`. Log the exception traceback information when an unhandled error occurs in a critical path.
*   **Constants**: Use \`ALL_CAPS_WITH_UNDERSCORES\` for global, unchanging constants.

### 3. External Rules
*Check configuration directories for explicit external rules:*
*   `.cursor/rules/*`: Contains rules specific to Cursor AI development environment.
*   `.github/copilot-instructions.md`: Contains guidelines for Copilot integration.