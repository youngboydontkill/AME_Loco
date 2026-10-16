#!/usr/bin/env bash
set -euo pipefail

AME_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${AME_ROOT}"
export PYTHONPATH="${AME_ROOT}:${AME_ROOT}/source/ame_locomotion:${PYTHONPATH:-}"

WAIT_FOR_RECOVERY_SUMMARY="${WAIT_FOR_RECOVERY_SUMMARY:?WAIT_FOR_RECOVERY_SUMMARY is required}"
QUEUE_LOG="${QUEUE_LOG:?QUEUE_LOG is required}"

while [[ ! -s "${WAIT_FOR_RECOVERY_SUMMARY}" ]]; do /usr/bin/sleep 30; done
echo "[$(date -Is)] recovery evaluation complete; running privileged-Critic smoke" >> "${QUEUE_LOG}"

PYTHONPATH="${AME_ROOT}:${AME_ROOT}/source/ame_locomotion:${PYTHONPATH:-}" \
  conda run --no-capture-output -n isaacsim5.1 python scripts/rsl_rl/train.py \
    --task AME-Kuavo-S54-Depth-privileged-critic --num_envs 2 --max_iterations 1 \
    --seed 42 --headless --enable_cameras \
    agent.experiment_name=appendix_privileged_critic_smoke agent.run_name=privileged_critic_smoke \
    >> "${QUEUE_LOG}" 2>&1
echo "[$(date -Is)] privileged-Critic smoke passed; starting baselines" >> "${QUEUE_LOG}"

GROUP_FILTER=cad,sensor_randomized,student_depth,student_privileged_critic,student_geometry \
SEEDS="42 43 44 45 46" EVAL_SEEDS="42 43 44 45 46" MAX_ITERATIONS=26000 \
NUM_ENVS=512 EVAL_EPISODES=32 EVAL_NUM_ENVS=64 \
  conda run --no-capture-output -n isaacsim5.1 bash scripts/rsl_rl/run_appendix_matrix.sh \
  >> "${QUEUE_LOG}" 2>&1
echo "[$(date -Is)] five non-confidence groups complete; evaluating remaining confidence training seeds" >> "${QUEUE_LOG}"

GROUP_FILTER=student_confidence SEEDS="42 43 44 45 46" EVAL_SEEDS="42 43 44 45 46" \
MAX_ITERATIONS=26000 NUM_ENVS=512 EVAL_EPISODES=32 EVAL_NUM_ENVS=64 \
  conda run --no-capture-output -n isaacsim5.1 bash scripts/rsl_rl/run_appendix_matrix.sh \
  >> "${QUEUE_LOG}" 2>&1
echo "[$(date -Is)] Appendix-A PPO matrix complete" >> "${QUEUE_LOG}"
