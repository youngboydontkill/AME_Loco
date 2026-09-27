# AME-Kuavo-S54-Depth-enhancement

该任务在 `AME-Kuavo-S54-Depth` 的 Isaac Lab 环境上，将相机的理想仿真深度先经过 Stable-Sim2Real Student，再交给原有 AME policy。机器人、相机、地形、奖励和动作空间保持不变，只有 policy/critic 的 `depth_image` 观测被替换。

```text
Isaac distance_to_image_plane [N, 42, 42]
    -> invalid depth sanitization (metres)
    -> frozen StudentNetwork (16 channels, 4 residual blocks)
    -> linear normalization to [-1, 1]
    -> existing ActorCriticEncoder
```

## Task names

- `AME-Kuavo-S54-Depth-enhancement`
- `AME-Kuavo-S54-Depth-enhancement-Play`

## Checkpoint

默认使用仓库内的 `pretrained/student_depth_latest.pt`。也可以在启动前覆盖：

```bash
export AME_STUDENT_CHECKPOINT=/absolute/path/to/student/latest.pt
```

checkpoint 应包含当前 stable-sim2real Student 的 `model_state_dict`，输入是 `[B, 1, 42, 42]` 的米制深度。Student 在环境初始化后的第一次观测调用时加载到 `env.device`，之后保持 eval/frozen 状态，不参与 PPO 反向传播。

## Training

```bash
python scripts/rsl_rl/train.py \
  --task AME-Kuavo-S54-Depth-enhancement \
  --num_envs 512 \
  --headless
```

Playback 使用：

```bash
python scripts/rsl_rl/play.py \
  --task AME-Kuavo-S54-Depth-enhancement-Play \
  --checkpoint /absolute/path/to/model_*.pt
```

## Important contract

原始 `depth_image` 使用线性 `0..5m -> -1..1` 归一化。增强任务保持这个输出契约，避免改变现有 `map_scan_dim=(42,42,1)` 与 Actor-Critic 输入维度。Student 内部使用米制输入；无效/NaN/Inf 相机像素按训练约定置为 `4.9m` 后再推理。

`pretrained/student_depth_latest.pt` 是从 stable-sim2real 当前 Student checkpoint 迁移的权重。它是推理预处理器，不是 RSL-RL 的 policy checkpoint；两者需要分别管理。
