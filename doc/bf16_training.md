# BF16 混合精度训练

AME 系列 PPO（含 S46、S54、Roban、深度、GRU、AMP-PPO 和学生微调）及
DAgger 配置默认启用 `agent.algorithm.use_mixed_precision=true`。
算法类的默认值仍为 `False`，兼容未增加该字段的旧配置。

在已安装本仓库两个 editable 包的 Isaac Lab 环境中运行：

```bash
python scripts/rsl_rl/train.py --task AME-G1-29DOF-v0 --headless agent.algorithm.use_mixed_precision=true
```

关闭时将末尾参数改成 `agent.algorithm.use_mixed_precision=false`。
请确认 `rsl_rl.__file__` 指向本仓库，避免使用其他项目的同名安装。

## 精度范围

- PPO minibatch 的策略、价值网络、辅助损失、对称损失和 RND 前向进入 BF16 autocast。
- 普通蒸馏与 DAgger 的学生前向、损失，以及 DAgger 的价值网络更新进入 autocast。
- rollout、GAE、教师（包括 PPO BC 教师）和 AMP 运动判别器保持原有 FP32 路径。
- 保留 Adam、FP32 主参数与优化器状态，无需 GradScaler，不引入 Muon。
- backward、梯度归约、裁剪和 step 在 autocast 外执行。反向算子可使用前向对应的低精度；
  FP32 参数的梯度保持 FP32。autocast 不保证所有前向算子都使用 BF16。
- checkpoint 权重格式不变，可使用原有 FP32 播放和导出路径。

CUDA 启用时检查原生 BF16 支持，不支持会提示关闭开关。
当前注意力请求 attention weights，使用显式注意力计算，未引入融合 SDPA；
因此没有移植 ame2_minimal 针对融合 BF16 注意力的 math backend 补丁。

## 验证

无需启动 Isaac Sim 的真实模型更新测试（CPU，并在可用时测试 CUDA）：

```bash
PYTHONPATH=rsl_rl python rsl_rl/tests/test_mixed_precision.py
```

覆盖注意力 PPO、GRU PPO、PPO+BC、普通蒸馏和 DAgger 的开关对照，检查
rollout 精度、实际 Linear 输出精度、参数更新、有限损失/梯度、FP32 参数和 Adam 状态，
以及冻结教师权重与 PPO BC 教师前向精度。

性能比较需固定任务、环境数、minibatch、种子和优化器，预热后分别测量更新时间、
整轮时间及峰值显存，并检查 KL 和奖励曲线。上述冒烟测试不替代长期收敛与仿真测速。
