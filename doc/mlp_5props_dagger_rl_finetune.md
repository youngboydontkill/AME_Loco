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
