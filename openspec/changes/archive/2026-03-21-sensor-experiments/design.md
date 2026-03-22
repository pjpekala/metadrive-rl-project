## Context

The current training pipeline uses `SafeMetaDriveEnv` with `LidarStateObservation` (259-dim vector, 240 lidar rays) and `MlpPolicy`. Training runs on CPU with an MLP trunk. The codebase has curriculum learning infrastructure with stage progression and evaluation callbacks.

MetaDrive provides multiple observation types:
1. **LiDAR**: 240 rays default, configurable up to 360+ rays
2. **TopDownMetaDriveEnvV2**: Bird's-eye view semantic map (84x84x5 channels: road_network, navigation, past_pos, traffic history)
3. **SafeMetaDriveEnv + RGBCamera**: Front-facing camera (64x64x3 stacked frames)

All camera-based observations return normalized float32 images, differing from SB3's default uint8 [0,255] expectation.

## Goals / Non-Goals

**Goals:**
- Add dense LiDAR, BEV, and RGB camera observation support for comparative experiments
- Use identical CNN architectures for fair visual modality comparison
- Minimize code duplication by sharing feature extractor implementation
- Maintain curriculum learning infrastructure for all new modalities
- Support GPU training for CNN policies
- Create unified evaluation framework across all sensor types

**Non-Goals:**
- Multi-sensor fusion experiments (future work)
- Modifying existing sparse LiDAR training pipeline
- Optimizing CNN architectures per modality separately
- Pretrained feature extractors (train from scratch)

## Decisions

### 1. Dense LiDAR Configuration

**Decision**: Add configurable lidar parameters to Stage dataclass

**Rationale**: MetaDrive's lidar supports configurable ray count and detection range through vehicle_config. Adding these to Stage enables curriculum-aware dense lidar experiments.

```python
@dataclass
class Stage:
    # ... existing fields ...
    use_dense_lidar: bool = False
    lidar_num_lasers: int = 360  # Default: 240
    lidar_distance: float = 50.0  # Default: 50
```

### 2. Custom CNN Feature Extractor

**Decision**: Create shared `CustomCNN` class extending `BaseFeaturesExtractor`

**Rationale**:
- SB3's `NatureCNN` expects 3-channel images and channel-first format
- MetaDrive returns channel-last images (H, W, C) and normalized float32
- Custom extractor handles permutation (H,W,C) → (C,H,W) internally

```python
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[-1]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
```

### 3. BEV Environment: TopDownMetaDriveEnvV2

**Decision**: Use `TopDownMetaDriveEnvV2` directly with custom wrapper

**Rationale**:
- Returns clean `(84, 84, 5)` Box space (no dict wrapping)
- Compatible with `CnnPolicy` + CustomCNN
- Built-in multi-channel: road_network, navigation, past_pos, traffic history
- `frame_stack=3`, `post_stack=5` provide temporal context

### 4. RGB Environment: SafeMetaDriveEnv + RGBCamera

**Decision**: Use `SafeMetaDriveEnv` with `image_observation=True` and custom RGBCamera

**Rationale**:
- Returns `{'image': (H,W,3,n), 'state': (19,)}` Dict space
- Requires `MultiInputPolicy` for handling dict observations
- CustomCNN for image branch, MLP for state branch
- `stack_size=1` to match BEV temporal channels

### 5. CNN Architecture Parity

**Decision**: Use identical CNN output dimension (256) and MLP trunk ([64, 64])

**Rationale**:
- Fair comparison: only input modality differs, not model capacity
- 8GB VRAM supports 512 batch size with this architecture
- ~900K CNN parameters per modality

### 6. Training Scripts: Separate Entry Points

**Decision**: Create `train_bev.py`, `train_rgb.py`, and `train_dense_lidar.py`

**Rationale**:
- Clean separation of experiments
- Different model directories per modality
- Independent training logs
- Easy to run comparison experiments

### 7. Unified Evaluation

**Decision**: Single `evaluate.py` with `--sensor` flag

**Rationalie**:
- `evaluate.py --sensor lidar` → loads LiDAR model
- `evaluate.py --sensor bev` → loads BEV model
- `evaluate.py --sensor rgb` → loads RGB model
- Auto-detects observation type from model directory

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| Training instability with CNN | CNN policies harder to train | Use same hyperparameters; start with curriculum |
| GPU memory limits | 8GB limits batch size | Use 64x64 RGB and 84x84 BEV |
| Dense LiDAR adds overhead | More rays = slower simulation | Use 360 rays (manageable on CPU) |
| VecNormalize with Dict | Dict obs with images | Test during implementation |

## Open Questions

1. **VecNormalize with RGB Dict**: Does VecNormalize properly normalize both image and state keys?

2. **Dense LiDAR baseline**: Should we train dense LiDAR from scratch or transfer from sparse?

3. **Experiment order**: Train all three concurrently or sequentially? (Sequential recommended)

## Module Structure

```
src/curriculum_learning/
├── feature_extractors.py    # NEW: CustomCNN
├── env.py                   # MODIFIED: add make_bev_env, make_rgb_env
├── config.py                # MODIFIED: add dense lidar config, ObservationType
├── train_bev.py             # NEW: BEV training
├── train_rgb.py             # NEW: RGB training
├── train_dense_lidar.py     # NEW: Dense LiDAR training (CPU, MlpPolicy)
└── evaluate.py              # MODIFIED: add --sensor flag
```
