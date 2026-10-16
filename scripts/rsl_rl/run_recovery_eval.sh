#!/usr/bin/env bash
set -euo pipefail

AME_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${AME_ROOT}"
export PYTHONPATH="${AME_ROOT}:${AME_ROOT}/source/ame_locomotion:${PYTHONPATH:-}"

RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
WAIT_FOR_SUMMARY="${WAIT_FOR_SUMMARY:?WAIT_FOR_SUMMARY is required}"
LOG_FILE="${LOG_FILE:?LOG_FILE is required}"
EPISODES="${EPISODES:-32}"
NUM_ENVS="${NUM_ENVS:-64}"
EVAL_SEEDS="${EVAL_SEEDS:-42 43 44 45 46}"
GROUP="${GROUP:-gemini_student_model26000_recovery}"

while [[ ! -s "${WAIT_FOR_SUMMARY}" ]]; do sleep 30; done
while pgrep -f '[t]rain.py.*resume12_from20000_to26k' >/dev/null; do sleep 10; done

mkdir -p logs/rsl_rl/evaluations
for seed in ${EVAL_SEEDS}; do
  output="logs/rsl_rl/evaluations/${GROUP}_seed${seed}.json"
  if [[ -s "${output}" ]]; then continue; fi
  PYTHONPATH="${AME_ROOT}:${AME_ROOT}/source/ame_locomotion:${PYTHONPATH:-}" \
  LD_LIBRARY_PATH="/home/hitcsc/miniconda3/envs/isaacsim5.1:/home/hitcsc/miniconda3/envs/isaacsim5.1/lib:${LD_LIBRARY_PATH:-}" \
  conda run --no-capture-output -n isaacsim5.1 python scripts/rsl_rl/evaluate_policy.py \
    --task AME-Kuavo-S54-Depth-enhancement-Play \
    --checkpoint "${CHECKPOINT}" \
    --episodes "${EPISODES}" --num_envs "${NUM_ENVS}" --seed "${seed}" \
    --group "${GROUP}" --output "${output}" \
    --recovery_eval --perturbation_step 100 --lateral_velocity_kick 0.8 \
    --recovery_error_threshold 0.35 --recovery_hold_seconds 0.5 --headless \
    >> "${LOG_FILE}" 2>&1
done

python scripts/rsl_rl/summarize_policy_evaluations.py \
  logs/rsl_rl/evaluations/${GROUP}_seed{42,43,44,45,46}.json \
  --output "logs/rsl_rl/evaluations/${GROUP}_seed_summary.json" >> "${LOG_FILE}" 2>&1
echo "recovery evaluation complete" >> "${LOG_FILE}"
