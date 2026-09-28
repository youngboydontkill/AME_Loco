# Student 深度图采集

`scripts/collect_student_depth.py` 在 Isaac Sim 中创建 S54 深度环境，加载 locomotion PPO 策略控制机器人，并同时保存相机原始深度和冻结 Student 的增强输出。默认使用训练该策略的原始深度任务 `AME-Kuavo-S54-Depth-v0`，避免把 Student 增强图误作为策略输入。默认会把 S54 的 8 类地形按 terrain-generator 列展开，因此一轮采集可以覆盖不同 terrain type 和多个难度 level。

默认策略 checkpoint 为 `logs/rsl_rl/kuavo_s54_depth_ame/2026-09-21_16-53-12/model_62000.pt`。脚本将 `base_velocity` 前进指令固定为 `0.5 m/s`（横向和转向指令为 0），每步用策略根据当前观测推理关节动作。可通过 `--policy_checkpoint` 更换策略，通过 `--forward_speed` 调整前进速度。这个 locomotion 策略负责让机器人运动；`pretrained/student_depth_latest.pt` 是另一个独立的冻结深度增强模型，不负责控制机器人。

在 `isaacsim5.1` 环境中运行：

```bash
cd /home/hitcsc/YX/AME_LCOO
source /home/hitcsc/miniconda3/etc/profile.d/conda.sh
conda activate isaacsim5.1

python scripts/collect_student_depth.py \
  --num_envs 64 \
  --steps 200 \
  --capture_interval 10 \
  --max_captures 5 \
  --headless \
  --output_dir outputs/student_depth
```

输出目录结构：

```text
outputs/student_depth/
  manifest.json
  env_0000/
    step_000010_raw.npy
    step_000010_raw_valid.npy
    step_000010_student.npy
    step_000010_difference.npy
    step_000010_comparison.png
```

数组是 `42x42` 的米制深度。`comparison.png` 从左到右为原始深度、Student 深度和 `Student - raw` 差值；原始相机的 NaN、Inf、越界像素在 `raw_valid.npy` 中标记为 false。`manifest.json` 记录每个样本的 env id、step、terrain type/name、terrain level、terrain origin 和文件路径。

默认的 `--all_terrains` 会覆盖所有地形类型。只想使用任务本身的 terrain 配置时，追加 `--no-all_terrains`；例如 Play 任务将使用其自身的单一 Play terrain：

```bash
python scripts/collect_student_depth.py \
  --task AME-Kuavo-S54-Depth-enhancement-Play \
  --no-all_terrains --num_envs 50 --headless
```

Student checkpoint 默认读取 `pretrained/student_depth_latest.pt`，也可以通过 `AME_STUDENT_CHECKPOINT` 覆盖。若要显式切换 Student 增强环境，可传入 `--task AME-Kuavo-S54-Depth-enhancement`；采集脚本仍单独运行上述 locomotion policy，并在保存数据时调用 Student。
