# MLP-5props Kuavo S54: DAgger 蒸馏与 PPO 微调

本文说明 `MLP-5props-Kuavo-Depth-Enhance-DAgger-v0` 到 `MLP-5props-Kuavo-Depth-Enhance-RL-v0` 的训练流程。DAgger 阶段让 student 学习冻结 teacher 的动作，同时用环境回报预训练 student critic；之后 PPO 微调从 DAgger checkpoint 初始化 student actor 和 critic。

## 流程

1. 用已训练的 teacher checkpoint 运行 DAgger task。teacher 固定参数，student 在 teacher 标注的状态上学习动作，并在学生控制的 rollout 上累积环境奖励。
2. 每次 DAgger rollout 后，以环境奖励、终止标记和 critic bootstrap 值计算 GAE target。行为克隆和速度辅助头使用 actor optimizer；critic 的 value loss 使用独立 optimizer。
3. DAgger checkpoint 保存完整 student/teacher 网络、actor optimizer、value optimizer 和训练迭代信息。
4. PPO 微调 task 读取 DAgger checkpoint 中的 student 深度编码器、props 编码器、actor、critic 和动作噪声参数。PPO optimizer 与训练状态重新初始化，微调迭代从头计数。

## 网络与观测

| 模块 | 输入 | 网络 | 训练用途 |
|---|---|---|---|
| Teacher actor | `teacher`：原始本体感知与 height scan | 沿用原 `ActorCriticEncoder`；terrain CNN、64 维 MHA（16 heads）、actor proprio 编码及 `[512, 256, 128]` actor | DAgger teacher 动作标签；参数冻结 |
| Student 深度编码器 | 一帧增强深度，`[batch, 42, 42]`（传入网络时展平为 1764） | Conv `1→16→32→64`，ELU 和下采样，Adaptive Average Pool，再经 Linear 得到 128 维特征 | DAgger 与 PPO 共用并迁移 |
| Student props 编码器 | 最近五帧本体感知，`[batch, 5, 90]`，展平为 450 维 | MLP `450→512→256→128` | DAgger 与 PPO 共用并迁移 |
| Student actor | 拼接深度和 props 特征，共 256 维 | MLP `[512, 256, 128]`，输出 27 维 Kuavo S54 动作 | DAgger 模仿 teacher；PPO 继续优化 |
| 速度辅助头 | Student props 的 128 维特征 | MLP，隐藏层 `[128]`，输出 3 维机体线速度 | DAgger 辅助监督，target 为 `teacher_critic[..., :3]`；PPO actor-critic 不使用此头 |
| Student critic | 完整 `teacher_critic` 观测组（含特权本体感知和 terrain scan） | MLP `[512, 256, 128]`，输出标量状态价值 | DAgger 以环境回报计算的 GAE target 单独训练；PPO 微调时初始化同一 critic |

`evaluate()` 保持用于查询 teacher 动作；student 价值估计使用独立的 `evaluate_value()`。DAgger 的 behavior/value 优化器互不包含对方网络参数。当前 DAgger 默认 `gamma=0.99`、`lambda=0.95`、value learning rate `1e-3`、每个 rollout 训练 1 个 value epoch；behavior loss 为 MSE，速度辅助 loss 系数为 `0.1`。

## 观测与环境配置

DAgger 和 PPO 微调使用相同的训练场景、增强深度和五帧 props：

- Student actor 输入：`depth`、`props`。
- 单帧 props 由角速度 3 维、投影重力 3 维、速度指令 3 维、关节位置 27 维、关节速度 27 维和上一时刻动作 27 维组成，共 90 维；props 张量形状是 `[batch, 5, 90]`。
- Teacher 动作输入：`teacher`。
- Student/PPO critic 输入：`teacher_critic`。
- 训练指令范围：前向速度 `0–1 m/s`，横向速度和角速度为 `0`。

环境奖励来自 29DOF 配置的 `RewardsCfg`，S54 配置会将接触、关节等 selector 映射到 Kuavo 部件；depth 环境把 `track_lin_vel_xy_exp` 的 `std` 设为 `0.5`。RL 微调有独立环境子类 `KuavoS54DepthCNNMLPRLEnvCfg`，可在其中覆盖 reward 权重或参数，不影响 DAgger 任务。例如：

```python
@configclass
class KuavoS54DepthCNNMLPRLEnvCfg(KuavoS54DepthCNNMLPDistillEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.4
```

PPO 默认沿用 S54 AME 配置：每次采样 24 steps/env，5 个 learning epochs、4 个 mini-batches、learning rate `1e-3`、`gamma=0.99`、`lambda=0.95`、entropy coefficient `0.008`；微调默认 20,000 iterations，每 500 iterations 保存一次。

## 训练与播放

先用 teacher checkpoint 训练 DAgger student：

```bash
python scripts/rsl_rl/train.py \
  --task MLP-5props-Kuavo-Depth-Enhance-DAgger-v0 \
  --teacher_checkpoint logs/rsl_rl/kuavo_s54_ame/2026-09-27_14-34-18/model_19998.pt \
  --max_iterations 20000 \
  --num_envs 1536 \
  --headles --enable_cameras
```

DAgger 日志默认保存在：

```text
logs/rsl_rl/kuavo_s54_depth_enhance_dagger_cnn_5props/<run目录>/model_<迭代>.pt
```

用 DAgger checkpoint 初始化 PPO 微调：

```bash
./isaac_lab23/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task MLP-5props-Kuavo-Depth-Enhance-RL-v0 \
  --student_checkpoint /path/to/dagger_model.pt \
  --headless
```

PPO 微调结果默认保存在 `logs/rsl_rl/kuavo_s54_depth_student_ppo_finetune/<run目录>/`。播放时将 `<ppo_checkpoint>` 替换为微调生成的 checkpoint：

```bash
./isaac_lab23/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task MLP-5props-Kuavo-Depth-Enhance-RL-Play-v0 \
  --checkpoint /path/to/ppo_checkpoint.pt
```

也可以用 `MLP-5props-Kuavo-Depth-Enhance-DAgger-Play-v0` 播放 DAgger checkpoint。

## Checkpoint 注意事项

- 新版 DAgger checkpoint 包含顶层 `critic.*` student 权重。PPO 微调加载器按网络名称和张量形状迁移深度编码器、props 编码器、actor、critic 与 `std`；teacher 和速度辅助头不会迁移到 PPO actor-critic。
- 增加 DAgger critic 之前训练的旧 checkpoint 不含 student `critic.*`。使用它微调时，actor 相关权重仍会加载，但 PPO critic 保持随机初始化。要从蒸馏 critic 开始微调，需用当前版本重新训练 DAgger。
- PPO 微调使用 `--student_checkpoint` 初始化 DAgger student。`--resume`/`--checkpoint` 用于恢复 PPO 微调本身的训练状态。

## 主要配置位置

- 任务注册：`source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/s54/__init__.py`
- DAgger 网络与算法参数：`source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/agents/kuavo_s54_depth_distill_cfg.py`
- PPO 微调网络与算法参数：`source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/agents/kuavo_s54_depth_rl_finetune_cfg.py`
- DAgger 环境与微调专用环境：`source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/s54/velocity_env_cfg_s54_depth_enhance.py`
- student/critic、GAE 与 checkpoint 迁移实现：`rsl_rl/rsl_rl/modules/student_teacher_cnn_mlp.py`、`rsl_rl/rsl_rl/algorithms/dagger.py`、`rsl_rl/rsl_rl/modules/actor_critic_cnn_props.py`


- physics_material
  - static_friction_range、dynamic_friction_range：刚体表面静摩擦和动摩擦系数的采样范围。摩擦较低时更容易打滑；摩擦变化范围增大时，策略需要适应更多地面接触条件。
  - restitution_range：恢复系数，控制碰撞弹性。范围增大可能让碰撞后的弹跳更明显。
  - num_buckets：离散的材料参数组合数。增加它会提供更多不同的摩擦/弹性组合；设为 1 时当前基本没有组合多样性。
  - asset_cfg：指定随机化应用到哪些刚体。现在指向机器人所有刚体 .*。
- add_base_mass
  - mass_distribution_params：对 waist_yaw_link 的质量增加量范围，单位是 kg。
  - operation="add"：在原质量上加减采样值，而不是用采样值替换原质量。增加范围有助于适应载荷变化，但范围过大可能导致惯性和控制响应明显改变。
- base_com
  - com_range.x/y/z：基座质心在各轴方向的偏移范围，单位是 m，作用于 waist_yaw_link。
  - 偏移会改变机器人受力和转动惯量的平衡，增加后能提升对载荷偏置的适应性；过大会让站立和行走更难。
- base_external_force_torque
  - force_range：reset 事件施加的力范围，单位 N。
  - torque_range：施加的力矩范围，单位 N·m。
  - 非零扰动可训练抗推能力；过大可能造成频繁摔倒。注意它的 mode="reset"，扰动发生在 reset 时，不是定时持续推搡。
- reset_base
  - pose_range.x/y：reset 时基座位置相对初始位置的偏移，单位 m。
  - pose_range.yaw：初始航向角偏移，单位 rad。
  - velocity_range.x/y/z：初始线速度范围，单位 m/s。
  - velocity_range.roll/pitch/yaw：初始角速度范围，单位 rad/s。
  - 增大这些范围会让策略从更多初始姿态和运动状态开始训练，增强恢复能力，但也会提高早期训练难度。
- reset_robot_joints
  - position_range：关节默认姿态的缩放因子范围；(1, 1) 表示不改变默认关节姿态。更宽的范围会让 reset 姿态偏离默认站姿。
  - velocity_range：关节初始速度范围，单位 rad/s。非零范围可让策略适应不同关节运动状态，但过大会导致 reset 后不稳定。
- push_robot
  - interval_range_s：两次 push 之间的时间范围，单位秒。
  - velocity_range.x/y：push 事件设置的水平速度范围，单位 m/s；它表示速度扰动，不是力的单位。增大后能训练抗扰动能力，但推力过强会增加摔倒和终止。
  - 当前范围为零，所以即使事件按间隔触发，也不会产生实际推搡。
- randomize_depth_camera
  - position_range.x/y/z：相机相对安装刚体局部位置的偏移范围，单位 m。
  - 非零偏移会改变相机视角和深度图内容，有利于适应安装误差；范围过大可能让训练图像与部署相机差异过大。