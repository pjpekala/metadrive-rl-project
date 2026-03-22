## ADDED Requirements

### Requirement: Dense LiDAR sensor configuration

The system SHALL provide configurable dense LiDAR sensor settings with customizable ray count and detection range.

#### Scenario: Stage configuration with dense LiDAR
- **WHEN** a Stage is created with `use_dense_lidar=True`
- **THEN** the stage SHALL include `lidar_num_lasers` (default: 360) and `lidar_distance` (default: 50.0)

#### Scenario: Dense LiDAR environment configuration
- **WHEN** `build_metadrive_config()` is called with a stage that has `use_dense_lidar=True`
- **THEN** the returned config SHALL include the dense lidar parameters in `vehicle_config['lidar']`

#### Scenario: Backward compatibility
- **WHEN** a Stage is created with `use_dense_lidar=False` (default)
- **THEN** the environment SHALL use default sparse LiDAR (240 rays, 50m range)

### Requirement: BEV observation training pipeline

The system SHALL provide a training pipeline for BEV (bird's-eye view) semantic map observations using TopDownMetaDriveEnvV2 and CnnPolicy.

#### Scenario: BEV environment returns correct observation shape
- **WHEN** BEV environment is created with default configuration
- **THEN** observation space SHALL be Box(-0.0, 1.0, (84, 84, 5), float32)
- **AND** reset() SHALL return numpy array of shape (84, 84, 5)

#### Scenario: BEV environment with curriculum stage
- **WHEN** BEV environment is reset with a curriculum stage configuration
- **THEN** environment SHALL use stage.map, stage.traffic_density, and stage.horizon

#### Scenario: CnnPolicy with custom CNN feature extractor
- **WHEN** PPO model is created with CnnPolicy and CustomCNN
- **THEN** policy SHALL accept (84, 84, 5) observation input
- **AND** CNN output dimension SHALL be 256

### Requirement: RGB camera observation training pipeline

The system SHALL provide a training pipeline for front-facing RGB camera observations using SafeMetaDriveEnv with RGBCamera and MultiInputPolicy.

#### Scenario: RGB environment returns correct observation structure
- **WHEN** RGB environment is created with default configuration
- **THEN** observation space SHALL be Dict with keys 'image' and 'state'
- **AND** image space SHALL be Box(-0.0, 1.0, (64, 64, 3, 1), float32)
- **AND** state space SHALL be Box(-0.0, 1.0, (19,), float32)

#### Scenario: RGB environment with curriculum stage
- **WHEN** RGB environment is reset with a curriculum stage configuration
- **THEN** environment SHALL use stage.map, stage.traffic_density, and stage.horizon

#### Scenario: MultiInputPolicy with CNN for image branch
- **WHEN** PPO model is created with MultiInputPolicy and CustomCNN for image
- **THEN** policy SHALL accept Dict observation with 'image' and 'state' keys
- **AND** image branch CNN output dimension SHALL be 256

### Requirement: Shared CNN architecture for fair comparison

The system SHALL use identical CNN architectures (same layer sizes, same output dimensions) for BEV and RGB experiments to ensure fair modality comparison.

#### Scenario: CNN architecture consistency
- **WHEN** BEV or RGB model is created
- **THEN** CNN SHALL have architecture: Conv2d(?, 32, 8x8, stride=4) → ReLU → Conv2d(32, 64, 4x4, stride=2) → ReLU → Conv2d(64, 64, 3x3, stride=1) → ReLU → Flatten → Linear(?, 256)
- **AND** CNN output dimension SHALL be 256

#### Scenario: MLP trunk consistency
- **WHEN** BEV or RGB model is created
- **THEN** MLP trunk SHALL use architecture [64, 64] for both policy and value networks

### Requirement: Training scripts for sensor experiments

The system SHALL provide separate training scripts for BEV, RGB, and dense LiDAR experiments that mirror the existing curriculum learning infrastructure.

#### Scenario: BEV training script
- **WHEN** train_bev.py is executed
- **THEN** script SHALL support curriculum progression through CONTINUOUS_CURRICULUM
- **AND** models SHALL be saved to models/bev/
- **AND** logs SHALL be saved to logs/bev/

#### Scenario: RGB training script
- **WHEN** train_rgb.py is executed
- **THEN** script SHALL support curriculum progression through CONTINUOUS_CURRICULUM
- **AND** models SHALL be saved to models/rgb/
- **AND** logs SHALL be saved to logs/rgb/

#### Scenario: Dense LiDAR training script
- **WHEN** train_dense_lidar.py is executed
- **THEN** script SHALL support curriculum progression through CONTINUOUS_CURRICULUM
- **AND** models SHALL be saved to models/dense_lidar/
- **AND** logs SHALL be saved to logs/dense_lidar/

### Requirement: GPU device support for visual sensors

The system SHALL support GPU training for BEV and RGB CNN policies using CUDA.

#### Scenario: CUDA device availability for visual sensors
- **WHEN** GPU is available
- **THEN** BEV and RGB training SHALL use device='cuda'

#### Scenario: CPU fallback for dense LiDAR
- **WHEN** dense LiDAR training is run
- **THEN** training SHALL use device='cpu' (MLP policy, no CNN)
