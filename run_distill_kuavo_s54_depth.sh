#!/bin/bash
set -euo pipefail

python scripts/rsl_rl/train.py \
  --task AME-Kuavo-S54-Depth-Enhance-DAgger-v0 \
  --teacher_checkpoint logs/rsl_rl/kuavo_s54_ame/2026-09-27_14-34-18/model_19998.pt \
  --max_iterations 20000 \
  --headless
