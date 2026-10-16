# AME-Kuavo-S54-Depth-enhancement

该任务在 `AME-Kuavo-S54-Depth` 的 Isaac Lab 环境上，将相机的理想仿真深度先经过 Stable-Sim2Real Student，再交给原有 AME policy。机器人、相机、地形、奖励和动作空间保持不变，policy/critic 的原单通道 `depth_image` map 扩展为增强深度与 confidence 两通道。

```text
Isaac distance_to_image_plane [N, 42, 42]
    -> invalid depth sanitization (metres)
    -> frozen StudentNetwork (32 channels, 6 residual blocks, depth + valid-mask input)
    -> [enhanced depth in [-1, 1], validity-limited confidence in [0, 1]]
    -> two-channel existing ActorCriticEncoder
```

## Task names

- `AME-Kuavo-S54-Depth-enhancement`
- `AME-Kuavo-S54-Depth-enhancement-Play`

## Checkpoint

默认使用仓库内的 `pretrained/student_depth_gemini_combined_20261005.pt`。该模型由
`student_data/gemini_stage2_combined` 的 2,161 个 scene 蒸馏训练，输入是
`[B, 2, 42, 42]` 的米制深度和相机有效掩码。旧的 confidence checkpoint 仍可通过环境变量显式指定：

```bash
export AME_STUDENT_CHECKPOINT=/absolute/path/to/student/latest.pt
```

checkpoint 包含冻结 Student 的 `model_state_dict` 和 architecture 元数据。Student 在环境初始化后的第一次观测调用时加载到 `env.device`，之后保持 eval/frozen 状态，不参与 PPO 反向传播。加载器也能从 state dict 推断并加载旧架构。

## Training

```bash
python scripts/rsl_rl/train.py \
  --task AME-Kuavo-S54-Depth-enhancement \
  --num_envs 64 \
  --enable_cameras \
  --headless
```

`run_train.sh` 使用同样的 64 个环境。该设置适配当前 16GB 显存的 RTX 5080；增大
`num_envs` 会显著增加相机和 PPO rollout 的显存占用。

Playback 使用：

```bash
python scripts/rsl_rl/play.py \
  --task AME-Kuavo-S54-Depth-enhancement-Play \
  --checkpoint /absolute/path/to/model_*.pt
```

## Observation contract

增强任务的 map observation 由一个 term 输出 `[N, 42*42*2]`。每个像素的两个相邻值依次为增强深度和 confidence，reshape 后对应 `[42,42,2]`，ActorCriticEncoder 使用 `map_scan_dim=(42,42,2)`。增强深度线性归一化到 `[-1,1]`，confidence 保持 `[0,1]`，并限制为 Student confidence 与相机有效掩码的逐像素最小值。Actor 与 Critic 共用该传感器观测；Student 内部使用米制输入，无效/NaN/Inf 相机像素按训练约定置为 `4.9m` 后推理，同时将有效掩码作为第二输入通道。

`mdp.student_depth_image` 仍返回旧的单通道深度格式，供采集与可视化工具使用。该 enhancement task 使用 `mdp.student_depth_confidence_image`，使旧工具和既有 depth-only task 保持原有接口。

带 confidence 的增强任务改变了 map observation 维度，策略网络形状与旧单通道 checkpoint 不兼容。开始带置信度策略训练时应新建 run，不要对旧 depth-only 策略 checkpoint 执行 resume。

`pretrained/student_depth_gemini_combined_20261005.pt` 是深度 Student 推理 checkpoint，不是 RSL-RL 的 policy checkpoint；两者需要分别管理。Gemini combined 训练集中的 confidence 监督 mask 全为有效，因此该 head 的有效像素输出未经校准；部署时 confidence 通道能表达相机已知的 invalid mask，但不能解释为深度误差方差。

几何头对照任务 `AME-Kuavo-S54-Depth-geometry-enhancement` 使用相同 Student checkpoint，map channels 为 `[depth, confidence, edge, foothold]`，对应 `map_scan_dim=(42,42,4)`；其 playback 任务名加 `-Play`。当前 combined 数据没有几何标签，所以 edge/foothold 只作为未校准的输出接口，不能解释为几何性能收益。

## Privileged Student variant

`AME-Kuavo-S54-Depth-enhancement-privileged` 保留增强深度给 Actor，同时把 Student 的
`[depth, validity, confidence, edge, foothold]` 五通道输出给 Critic。Actor 使用
`actor_map_scan_dim=(42,42,1)`，Critic 使用 `critic_map_scan_dim=(42,42,5)`；对应的 playback
任务名加 `-Play`。该任务使用非对称 Actor/Critic 编码器，Critic 的辅助图只用于价值估计。

所有使用 Student observation 的训练任务默认按并行环境独立注入 20% 的随机化原始仿真深度，
其余环境使用 Student 增强结果。随机化包括高斯深度噪声和像素 dropout；当 Critic 处于原始深度
分支时，辅助先验通道置零以保持固定输入形状。比例可通过 observation term 的
`raw_depth_probability` 参数调整。
