## 1. Infrastructure Foundation

- [x] 1.1 Create `src/curriculum_learning/feature_extractors.py` with CustomCNN class
- [x] 1.2 Add CustomCNN as BaseFeaturesExtractor with channel-last to channel-first permutation
- [x] 1.3 Add ObservationType enum to `src/curriculum_learning/config.py` (LIDAR, DENSE_LIDAR, BEV, RGB)
- [x] 1.4 Add dense LiDAR fields to Stage dataclass: `use_dense_lidar`, `lidar_num_lasers`, `lidar_distance`
- [x] 1.5 Add BEVSensorConfig and RGBSensorConfig dataclasses to config.py
- [x] 1.6 Add cnn_features_dim and policy fields to TrainingConfig dataclass

## 2. Dense LiDAR Environment Integration

- [x] 2.1 Modify `build_metadrive_config()` in env.py to pass dense lidar parameters from stage
- [x] 2.2 Verify backward compatibility when `use_dense_lidar=False`
- [x] 2.3 Test dense LiDAR observation space with 360 rays

## 3. BEV Environment Integration

- [x] 3.1 Add `build_bev_config()` function to `src/curriculum_learning/env.py`
- [x] 3.2 Create `BEVCurriculumEnv` class extending TopDownMetaDriveEnvV2
- [x] 3.3 Add `make_bev_env()` factory function to env.py
- [x] 3.4 Add `make_bev_inline_eval_env()` factory function to env.py
- [x] 3.5 Wrap BEV envs with Monitor for logging

## 4. RGB Environment Integration

- [x] 4.1 Add `build_rgb_config()` function to env.py with RGBCamera setup
- [x] 4.2 Create `RGBCurriculumEnv` class extending SafeMetaDriveEnv
- [x] 4.3 Add `make_rgb_env()` factory function to env.py
- [x] 4.4 Add `make_rgb_inline_eval_env()` factory function to env.py
- [x] 4.5 Wrap RGB envs with Monitor for logging

## 5. BEV Training Script

- [x] 5.1 Create `src/curriculum_learning/train_bev.py`
- [x] 5.2 Add `run_bev_training()` function with CnnPolicy + CustomCNN
- [x] 5.3 Configure model output to `models/bev/` and logs to `logs/bev/`
- [x] 5.4 Add BEV curriculum callbacks (reuse CurriculumCallback)
- [x] 5.5 Add `run_bev()` entry point function
- [x] 5.6 Update Justfile with `train bev` command

## 6. RGB Training Script

- [x] 6.1 Create `src/curriculum_learning/train_rgb.py`
- [x] 6.2 Add `run_rgb_training()` function with MultiInputPolicy + CustomCNN
- [x] 6.3 Configure model output to `models/rgb/` and logs to `logs/rgb/`
- [x] 6.4 Add RGB curriculum callbacks (reuse CurriculumCallback)
- [x] 6.5 Add `run_rgb()` entry point function
- [x] 6.6 Update Justfile with `train rgb` command

## 7. Dense LiDAR Training Script

- [x] 7.1 Create `src/curriculum_learning/train_dense_lidar.py`
- [x] 7.2 Add `run_dense_lidar_training()` function with MlpPolicy
- [x] 7.3 Configure model output to `models/dense_lidar/` and logs to `logs/dense_lidar/`
- [x] 7.4 Add dense LiDAR curriculum callbacks (reuse CurriculumCallback)
- [x] 7.5 Add `run_dense_lidar()` entry point function
- [x] 7.6 Update Justfile with `train dense-lidar` command

## 8. Evaluation Updates

- [x] 8.1 Add `--sensor` argument to `src/curriculum_learning/evaluate.py`
- [x] 8.2 Auto-detect observation type from model path
- [x] 8.3 Add `evaluate_bev()` function
- [x] 8.4 Add `evaluate_rgb()` function
- [x] 8.5 Add `evaluate_dense_lidar()` function
- [x] 8.6 Update Justfile eval commands to support --sensor flag

## 9. Testing and Verification

- [x] 9.1 Test dense LiDAR environment creation and observation shape
- [x] 9.2 Test BEV environment creation and observation shape
- [x] 9.3 Test RGB environment creation and observation shape
- [x] 9.4 Test CnnPolicy with CustomCNN for BEV (single env, no training)
- [x] 9.5 Test MultiInputPolicy with CustomCNN for RGB (single env, no training)
- [x] 9.6 Test VecNormalize with BEV observations
- [x] 9.7 Test VecNormalize with RGB Dict observations
- [x] 9.8 Run linter on new files
