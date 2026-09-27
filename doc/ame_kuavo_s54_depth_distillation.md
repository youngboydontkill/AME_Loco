# AME-Kuavo-S54 depth distillation

新增任务 `AME-Kuavo-S54-Depth-Distill-v0` 使用 42x42 depth camera 作为 student 输入，同时在仿真中保留原 S54 33x21x3 height-scan 作为 privileged teacher 输入。teacher 网络被冻结，训练目标是 teacher actor 输出的行为克隆损失，student 结构仍使用 AME 的 CNN + Multi-Head Attention encoder。

先安装本地包：

```bash
python -m pip install -e source/ame_locomotion
python -m pip install -e rsl_rl
```

使用已有 height-scan PPO checkpoint 启动蒸馏：

```bash
python scripts/rsl_rl/train.py \
  --task AME-Kuavo-S54-Depth-Distill-v0 \
  --teacher_checkpoint logs/rsl_rl/kuavo_s54_ame/<run>/model_<step>.pt \
  --headless
```

`--teacher_checkpoint` 只用于首次初始化 teacher。蒸馏运行会写入 `logs/rsl_rl/kuavo_s54_depth_distill/`；后续可使用普通 `--resume --load_run ... --checkpoint ...` 继续训练。播放时使用 `AME-Kuavo-S54-Depth-Distill-Play-v0` 和蒸馏生成的 checkpoint，推理只走 student 的 depth 输入。

原有 `AME-Kuavo-S54-Depth-v0`、height-scan 任务和对应配置未被替换。
