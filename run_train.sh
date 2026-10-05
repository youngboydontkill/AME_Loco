#!/bin/bash
python scripts/rsl_rl/train.py \
--task AME-Kuavo-S54-Depth-enhancement \
--num_envs 64 \
--max_iterations 15000 \
--enable_cameras \
--headless
