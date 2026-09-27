# Kuavo S54 Depth AME 实验总结

## 1. 实验概况

| 项目 | 内容 |
| --- | --- |
| 实验名称 | `kuavo_s54_depth_ame` |
| 任务 | Kuavo S54 深度视觉 AME locomotion |
| 日志目录 | `logs/rsl_rl/kuavo_s54_depth_ame/2026-08-24_20-22-05` |
| 训练开始时间 | 2026-08-24 20:23:16 |
| 日志最后时间 | 2026-08-26 17:47:27 |
| 训练日志步数 | 38,775 次迭代记录 |
| 计划最大迭代 | 40,000 |
| 最后保存模型 | `model_38000.pt` |
| 随机种子 | 42 |
| 设备 | `cuda:0` |
| 并行环境 | 512 |
| 每环境采样步数 | 24 |
| 估算采样量 | 约 476.5M transitions（38,775 × 512 × 24） |
| 日志器 | TensorBoard |
| 独立评估 | 未发现独立 play/evaluation 日志；以下结果均为训练期统计 |

本报告以实验目录中的 `params/agent.yaml`、`params/env.yaml`、TensorBoard
事件文件和 `git/AME_LCOO.diff` 为准。日志快照记录的 `max_iterations=40000`、
`save_interval=1000` 与当前源码中的同名配置可能不同，不能用当前源码配置替代
本次实验的实际配置。

## 2. 环境与仿真配置

### 2.1 机器人与执行器

- 机器人资产：`kuavo/assets/biped_s54/urdf/biped_s54.urdf`。
- 27 个可控关节：双腿 12 DoF、腰部 yaw 1 DoF、双臂 14 DoF。
- 初始根部位置：`(0, 0, 0.925)`，初始线速度和角速度均为 0。
- 初始腿部姿态：
  - `leg_[lr]1_joint=0.0`
  - `leg_[lr]2_joint=0.0`
  - `leg_[lr]3_joint=-0.4`
  - `leg_[lr]4_joint=0.69`
  - `leg_[lr]5_joint=-0.33`
  - `leg_[lr]6_joint=0.0`
  - 腰部和双臂关节为 0。
- 执行器类型：`DelayedPDActuator`。
- 控制命令延迟：0～4 个 physics step。
- 主要 PD 参数：

| 关节组 | stiffness | damping |
| --- | ---: | ---: |
| 腿部 1/2 | 60 | 6 |
| 腿部 3 | 80 | 6 |
| 腿部 4 | 95 | 6 |
| 腿部 5/6 | 55 | 7.5 |
| 腰部 yaw | 40 | 4 |
| 手臂 1～4 | 20 | 3 |
| 手臂 5～7 | 15 | 3 |

### 2.2 仿真参数

- physics timestep：`dt=0.005 s`。
- 控制 decimation：4，因此策略控制周期约为 `0.02 s`。
- 渲染间隔：4。
- 重力：`(0, 0, -9.81)`。
- 物理材质默认摩擦系数：静摩擦 1.0、动摩擦 1.0、恢复系数 0。
- 机器人启用 self-collision。
- articulation solver iterations：position 8、velocity 4。
- 碰撞类型：`convex_hull`。
- 并行场景：512 环境，环境间距 2.5 m。
- 接触传感器更新周期：0.005 s，历史长度 3，接触阈值 1.0。

### 2.3 地形

- 地形类型：程序生成的 rough terrain。
- 课程学习：开启，10 行 × 20 列。
- 单块地形尺寸：8 m × 8 m。
- 难度范围：0～1。
- 初始地形等级：0。
- 地形采样比例：

| 地形 | 比例 |
| --- | ---: |
| pyramid stairs | 10% |
| inverted pyramid stairs | 10% |
| random boxes | 10% |
| random rough | 10% |
| positive pyramid slope | 10% |
| inverted pyramid slope | 10% |
| stepping stones | 20% |
| concentric gaps | 20% |

## 3. 观测、动作与深度相机

### 3.1 Policy 观测

Policy 使用带 corruption 的观测组，包含：

- base angular velocity，scale `0.2`；
- projected gravity；
- base velocity command；
- 相对关节位置；
- 相对关节速度，scale `0.05`；
- 上一步动作；
- 深度图 `depth_image`；
- height scan 未使用。

Critic 观测与 policy 基本一致，但额外包含真实 `base_lin_vel`，且
`enable_corruption=false`。Policy 和 critic 均采用 `ActorCriticEncoder`，
深度输入维度为 `(42, 42, 1)`，启用 CNN downsample。

### 3.2 动作

- 动作类型：关节位置动作。
- 匹配关节：全部关节。
- action scale：`0.25`。
- offset：`0.0`。
- 使用默认关节位置作为 offset：开启。

### 3.3 深度相机

- 挂载位置：`waist_yaw_link/depth_camera`。
- 更新周期：`0.02 s`。
- 分辨率：42 × 42，单通道。
- 数据类型：`distance_to_image_plane`。
- 投影：pinhole。
- clipping range：0.05～5.0 m。
- focal length：15.5335。
- 水平 aperture：20.955。
- 相机姿态锁定：开启。

## 4. 随机化、重置与任务目标

- 启动时随机化机器人各刚体物理材质：
  - 静摩擦：0.3～1.0；
  - 动摩擦：0.3～1.0；
  - 恢复系数：0.0～0.1；
  - 64 个材质 bucket。
- 根部 reset：
  - x、y 位置分别在 -0.5～0.5 m 内随机；
  - yaw 在 -3.14～3.14 rad 内随机；
  - 初始速度全部为 0。
- 关节 reset：位置按默认姿态 scale=1.0，关节速度为 0。
- `push_robot=null`，未启用周期性推扰。
- episode 时长：20 s。
- 速度指令每 10 s 重采样一次：
  - x 方向线速度：0～1.5 m/s；
  - y 方向线速度：0；
  - yaw 角速度：-1～1 rad/s；
  - heading command：开启，heading 控制刚度 0.5；
  - standing 环境比例：0。

## 5. Reward 与终止条件

### 5.1 Reward 权重

| Reward 项 | 权重 |
| --- | ---: |
| termination penalty | -200 |
| track linear velocity | 2.0 |
| track yaw velocity | 3.0 |
| angular velocity xy | -0.05 |
| undesired contacts | -1.0 |
| joint torques | -1.5e-7 |
| joint accelerations | -1.25e-7 |
| joint velocities | -0.001 |
| joint position limits | -1.0 |
| torque limits | -0.01 |
| action rate | -0.01 |
| flat orientation | -2.0 |
| feet air time | 0.25 |
| feet air-time variance | -0.7 |
| feet slide | -0.1 |
| feet stumble | -2.0 |
| feet too near | -1.0 |
| joint coordination | -0.2 |
| hip deviation | -0.1 |
| arm deviation | -0.3 |
| waist deviation | -1.0 |

线速度跟踪 reward 的指数核 `std=0.5`，yaw 速度跟踪 reward 的
`std=0.25`。足端相关 reward 使用 `leg_[lr]6_link`。

### 5.2 终止与课程

- `time_out`：episode 到达 20 s，属于正常结束。
- `base_contact`：`base_link` 或 `waist_yaw_link` 接触，阈值 1.0，属于失败终止。
- 启用了 `terrain_progress_diagnostics`，记录通过距离、要求距离、上下调级比例
  和平均地形等级。
- 本实验从等级 0 开始，前期主要在等级 0 学习基础行走，约 25.4k 次迭代后
  才开始出现地形升级。

## 6. PPO 与网络配置

### 6.1 Runner

- `num_steps_per_env=24`
- `max_iterations=40000`
- `save_interval=1000`
- `resume=false`
- `load_run=.*`
- `load_checkpoint=model_.*.pt`
- `seed=42`

### 6.2 Actor-Critic

- 类：`ActorCriticEncoder`
- actor hidden dims：`[512, 256, 128]`
- critic hidden dims：`[512, 256, 128]`
- activation：ELU
- 初始 action noise std：1.0
- actor/critic observation normalization：关闭
- 深度 map shape：`42 × 42 × 1`
- CNN downsample：开启

### 6.3 PPO 算法

- learning rate：`1e-3`
- schedule：adaptive
- learning epochs：5
- mini-batches：4
- gamma：0.99
- GAE lambda：0.95
- entropy coefficient：0.008
- desired KL：0.01
- max grad norm：1.0
- value loss coefficient：1.0
- clipped value loss：开启
- clip parameter：0.2
- RND：关闭
- symmetry augmentation：关闭

## 7. 训练结果

### 7.1 分阶段统计

下表为 TensorBoard 指标在指定迭代区间的平均值。`base_contact` 和
`time_out` 是对应终止原因在批次中的比例。

| 迭代区间 | 平均回报 | 线速度误差 | yaw 速度误差 | timeout | base contact | 平均地形等级 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0～5k | -2.56 | 0.608 | 0.331 | 42.2% | 57.8% | 0.001 |
| 5k～10k | 5.39 | 0.854 | 0.245 | 80.0% | 20.0% | 0 |
| 10k～15k | 18.01 | 1.319 | 0.295 | 87.5% | 12.5% | 0 |
| 15k～20k | 25.11 | 1.379 | 0.268 | 90.5% | 9.5% | 0 |
| 20k～25k | 26.84 | 1.359 | 0.262 | 89.5% | 10.5% | 0 |
| 25k～30k | 24.64 | 1.322 | 0.279 | 84.6% | 15.5% | 0.001 |
| 30k～33k | 18.62 | 1.015 | 0.325 | 67.1% | 32.9% | 0.82 |
| 33k～35k | 3.93 | 0.328 | 0.446 | 36.8% | 63.2% | 4.38 |
| 35k～37k | 2.30 | 0.312 | 0.513 | 41.5% | 58.5% | 4.56 |
| 37k～38.775k | 3.55 | 0.305 | 0.526 | 47.8% | 52.2% | 4.57 |

### 7.2 关键峰值与最终状态

| 指标 | 最佳/关键值 | 迭代 |
| --- | ---: | ---: |
| 最大单点平均回报 | 30.79 | 23,455 |
| 20k 附近平均回报 | 约 29.8 | 20,000 |
| 最终平均回报 | 4.35 | 38,774 |
| 最终平均 episode length | 701.75 steps | 38,774 |
| 最终线速度误差 | 0.392 | 38,774 |
| 最终 yaw 速度误差 | 0.688 | 38,774 |
| 最终 timeout 比例 | 49.2% | 38,774 |
| 最终 base-contact 比例 | 50.8% | 38,774 |
| 最终平均地形等级 | 4.79 | 38,774 |
| 训练期间最高平均地形等级 | 7.02 | 38,470 |

训练在 15k～25k 迭代期间达到相对稳定的平地行走状态：平均回报约
25～27，timeout 比例约 90%，base-contact 比例约 10%。从约 25.4k 迭代开始，
地形课程开始升级；进入等级 4～7 后，平均回报明显下降，base-contact 终止
比例升高到约 50%～60%。

### 7.3 PPO 训练信号

- value loss 在平地训练中期约 0.04～0.06，进入地形升级阶段后升至约
  0.10～0.16，说明价值估计面对更强的地形分布变化。
- entropy 在中期最低约 33.6，末期上升到约 46.5。
- policy mean noise std 从约 1.0～1.1 上升到末期约 1.47。
- 自适应学习率末期仍在 `1e-5`～`7.6e-5` 范围内变化。
- 末期线速度误差约 0.30～0.39，但 yaw 速度误差约 0.53～0.69。
  因此末期较低的线速度误差不能单独代表策略整体性能良好，因为同时存在
  较高的跌倒终止率和较差的 yaw 跟踪。

## 8. 实验中采用的代码修改

实验目录中的 `git/AME_LCOO.diff` 记录了以下改动：

1. 增加 `terrain_progress_diagnostics`，用于记录课程升级前后的距离和调级比例。
2. S54 depth 任务从地形等级 0 启动。
3. 禁用 depth bootstrap 阶段的 reset 速度扰动和 `push_robot`。
4. 将线速度跟踪 reward 的 `std` 调整为 0.5。
5. 将 S54 的 base-contact 终止关注范围收窄到真正的 base/waist link。
6. 将场景并行数相关配置从 2048 调整到 1024 的改动出现在同一 diff 中，
   但本次实际日志快照使用的是 512 个并行环境。

## 9. 结论

本实验证明了 S54 深度视觉 AME 策略能够在等级 0 平地环境上学习到基本的
速度跟踪和稳定行走能力，约 20k～25k 迭代达到训练期最佳表现。课程学习
开始后，策略进入更高等级地形时出现明显性能退化：回报从约 25～30 降至
约 2～5，base-contact 终止率从约 10% 上升至约 50% 以上。

因此，本次实验的主要结果应表述为：

- 平地基础 locomotion：已学会，训练期表现较好；
- 深度视觉输入链路：已参与训练，未出现日志层面的输入或运行失败；
- 高难度地形泛化：当前不稳定，尚未达到可用于最终部署的程度；
- 推荐使用：优先检查 `model_20000.pt`～`model_25000.pt` 的平地表现，
  不建议直接把 `model_38000.pt` 作为最终模型。

## 10. 后续建议

1. 对 `model_20000.pt`、`model_23000.pt` 和 `model_38000.pt` 分别运行固定种子、
   固定地形等级的 play 评估，区分平地能力和课程地形能力。
2. 记录每个 terrain level 的成功率、跌倒率、线速度误差和 yaw 误差，而不是
   只看全局平均回报。
3. 放缓 terrain curriculum，或在进入等级 1～3 后增加保持时间，避免策略从
   等级 0 直接快速进入尚未掌握的地形。
4. 检查课程升级后的探索噪声增长、yaw 跟踪退化和 base-contact 增长之间的关系。
5. 将本实验的 TensorBoard 曲线与后续 S54 depth run 对比，确认退化是课程策略、
   相机输入、奖励比例还是随机种子导致。

## 11. 数据来源

- `logs/rsl_rl/kuavo_s54_depth_ame/2026-08-24_20-22-05/params/agent.yaml`
- `logs/rsl_rl/kuavo_s54_depth_ame/2026-08-24_20-22-05/params/env.yaml`
- `logs/rsl_rl/kuavo_s54_depth_ame/2026-08-24_20-22-05/events.out.tfevents.1787574190.hitcsc.1261119.0`
- `logs/rsl_rl/kuavo_s54_depth_ame/2026-08-24_20-22-05/git/AME_LCOO.diff`
- `logs/rsl_rl/kuavo_s54_depth_ame/2026-08-24_20-22-05/model_*.pt`
