## Why

The current training uses sparse LiDAR (240 rays) with an MLP policy. To explore how perception modality affects driving policy learning, we need to add support for: (1) dense LiDAR with more rays and range, (2) bird's-eye view (BEV) semantic map, and (3) front-facing RGB camera. These experiments will help understand whether spatial structure (BEV), visual appearance (RGB), or enhanced point density (dense LiDAR) influences learning efficiency and final performance.

## What Changes

- **Dense LiDAR Configuration**: Add configurable lidar ray count (up to 360) and detection range to Stage config
- **BEV Observation Pipeline**: Add TopDownMetaDriveEnvV2 with BEV semantic map (84x84x5) and CnnPolicy
- **RGB Camera Observation Pipeline**: Add SafeMetaDriveEnv with RGBCamera (64x64x3) and MultiInputPolicy
- **Shared CustomCNN Feature Extractor**: Common architecture for both BEV and RGB with identical output dimensions for fair comparison
- **Unified Evaluation Framework**: Evaluate models across all three sensor modalities

## Capabilities

### New Capabilities

- `sensor-experiments`: Support for dense LiDAR, BEV, and RGB camera observation modalities for comparative experiments
- `dense-lidar-sensor`: Dense LiDAR sensor configuration with configurable ray count and detection range
- `bev-observation`: Bird's-eye view semantic map observation using MetaDrive's TopDownMultiChannel renderer
- `rgb-observation`: Front-facing RGB camera observation with configurable resolution and frame stacking
- `cnn-feature-extractor`: Custom CNN feature extractor for Box and Dict observation spaces
- `sensor-comparison-experiment`: Framework for comparing sparse LiDAR, dense LiDAR, BEV, and RGB on learning performance

### Modified Capabilities

- (none)

## Impact

- **New modules**: `feature_extractors.py`, `train_bev.py`, `train_rgb.py`, `train_dense_lidar.py`
- **Modified modules**: `config.py`, `env.py`, `evaluate.py`
- **Dependencies**: stable-baselines3 (PPO/CnnPolicy/MultiInputPolicy), metadrive
- **Training resources**: GPU required for BEV/RGB CNN policies; CPU sufficient for dense LiDAR
- **Model outputs**: `models/sparse_lidar/`, `models/dense_lidar/`, `models/bev/`, `models/rgb/`
