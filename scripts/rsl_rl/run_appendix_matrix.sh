#!/usr/bin/env bash
set -euo pipefail

AME_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${AME_ROOT}"
export PYTHONPATH="${AME_ROOT}:${AME_ROOT}/source/ame_locomotion:${PYTHONPATH:-}"

# IsaacSim 5.1 needs the environment's libstdc++ for CXXABI_1.3.15.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}:${LD_LIBRARY_PATH:-}"
fi

# Uniform Appendix-A PPO matrix. Each group uses the same terrain, rollout,
# seed list, and iteration budget; only the registered observation task changes.
NUM_ENVS="${NUM_ENVS:-512}"
MAX_ITERATIONS="${MAX_ITERATIONS:-26000}"
FINAL_STEP="${FINAL_STEP:-$((MAX_ITERATIONS - 1))}"
SEEDS="${SEEDS:-42 43 44 45 46}"
GROUP_FILTER="${GROUP_FILTER:-all}"
LOG_ROOT="${LOG_ROOT:-logs/rsl_rl}"
EVAL_SEEDS="${EVAL_SEEDS:-42 43 44 45 46}"
EVAL_EPISODES="${EVAL_EPISODES:-32}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-64}"
HEADLESS_ARGS=(--headless --enable_cameras)

declare -a MATRIX_GROUPS=(
  "cad|AME-Kuavo-S54-Depth-v0"
  "sensor_randomized|AME-Kuavo-S54-Depth-sensor-randomized"
  "student_depth|AME-Kuavo-S54-Depth-student"
  "student_confidence|AME-Kuavo-S54-Depth-enhancement"
  "student_privileged_critic|AME-Kuavo-S54-Depth-privileged-critic"
  "student_geometry|AME-Kuavo-S54-Depth-geometry-enhancement"
)

for group_spec in "${MATRIX_GROUPS[@]}"; do
  IFS='|' read -r group task <<< "${group_spec}"
  if [[ "${GROUP_FILTER}" != "all" && ",${GROUP_FILTER}," != *",${group},"* ]]; then
    continue
  fi
  for seed in ${SEEDS}; do
    run_name="${group}_seed${seed}_${MAX_ITERATIONS}"
    experiment_dir="${LOG_ROOT}/appendix_${group}"
    latest_checkpoint=""
    latest_step=0
    if [[ -d "${experiment_dir}" ]]; then
      while IFS= read -r run_dir; do
        while IFS= read -r checkpoint; do
          step="${checkpoint##*/model_}"
          step="${step%.pt}"
          if [[ "${step}" =~ ^[0-9]+$ ]] && (( step > latest_step )); then
            latest_step="${step}"
            latest_checkpoint="${checkpoint}"
          fi
        done < <(find "${run_dir}" -maxdepth 1 -type f -name 'model_*.pt' -print 2>/dev/null)
      done < <(find "${experiment_dir}" -mindepth 1 -maxdepth 1 -type d -name "*_${run_name}" -print 2>/dev/null)
    fi

    if (( latest_step < FINAL_STEP )); then
      resume_args=()
      remaining_iterations="${MAX_ITERATIONS}"
      if [[ -n "${latest_checkpoint}" ]]; then
        remaining_iterations="$((MAX_ITERATIONS - latest_step))"
        load_run="$(basename "$(dirname "${latest_checkpoint}")")"
        resume_args=(--resume --load_run "${load_run}" --checkpoint "$(basename "${latest_checkpoint}")")
        echo "[appendix] resume group=${group} seed=${seed} from=${latest_step} remaining=${remaining_iterations}"
      else
        echo "[appendix] start group=${group} task=${task} seed=${seed} target=${MAX_ITERATIONS}"
      fi

      python scripts/rsl_rl/train.py \
        --task "${task}" \
        --num_envs "${NUM_ENVS}" \
        --max_iterations "${remaining_iterations}" \
        --seed "${seed}" \
        "${resume_args[@]}" \
        "${HEADLESS_ARGS[@]}" \
        agent.experiment_name="appendix_${group}" \
        agent.run_name="${run_name}" \
        agent.save_interval=500
    else
      echo "[appendix] training complete group=${group} seed=${seed} step=${latest_step}"
    fi

    latest_checkpoint=""
    latest_step=0
    # RSL-RL can return just before the final torch.save() becomes visible.
    # Rescan briefly so a completed training run is not misclassified as
    # missing its final checkpoint.
    for checkpoint_attempt in $(seq 0 60); do
      latest_checkpoint=""
      latest_step=0
      while IFS= read -r run_dir; do
        while IFS= read -r checkpoint; do
          step="${checkpoint##*/model_}"
          step="${step%.pt}"
          if [[ "${step}" =~ ^[0-9]+$ ]] && (( step > latest_step )); then
            latest_step="${step}"
            latest_checkpoint="${checkpoint}"
          fi
        done < <(find "${run_dir}" -maxdepth 1 -type f -name 'model_*.pt' -print 2>/dev/null)
      done < <(find "${experiment_dir}" -mindepth 1 -maxdepth 1 -type d -name "*_${run_name}" -print 2>/dev/null)
      (( latest_step >= FINAL_STEP )) && break
      /usr/bin/sleep 2
    done
    # RSL-RL saves the last iteration as MAX_ITERATIONS-1. Create a stable
    # model_MAX_ITERATIONS alias so downstream reports use one checkpoint name.
    if (( latest_step == FINAL_STEP )) && [[ "${latest_step}" != "${MAX_ITERATIONS}" ]]; then
      final_alias="$(dirname "${latest_checkpoint}")/model_${MAX_ITERATIONS}.pt"
      if [[ ! -e "${final_alias}" ]]; then
        ln "${latest_checkpoint}" "${final_alias}"
      fi
      latest_checkpoint="${final_alias}"
      latest_step="${MAX_ITERATIONS}"
    fi
    if (( latest_step < MAX_ITERATIONS )) || [[ -z "${latest_checkpoint}" ]]; then
      echo "[appendix] error: final checkpoint missing for ${group} seed ${seed}" >&2
      exit 1
    fi

    eval_dir="${LOG_ROOT}/evaluations"
    eval_reports=()
    if [[ "${task}" == *-v0 ]]; then
      play_task="${task%-v0}-Play-v0"
    else
      play_task="${task}-Play"
    fi
    for eval_seed in ${EVAL_SEEDS}; do
      report="${eval_dir}/appendix_${group}_train${seed}_eval${eval_seed}.json"
      report_valid=0
      if [[ -s "${report}" ]]; then
        if python - "${report}" "${latest_checkpoint}" "${play_task}" "appendix_${group}_train${seed}" "${eval_seed}" "${EVAL_EPISODES}" "${EVAL_NUM_ENVS}" <<'PY'
import json
import sys
from pathlib import Path

report_path, checkpoint, task, group, seed, episodes, num_envs = sys.argv[1:]
try:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    valid = (
        Path(report["checkpoint"]).resolve() == Path(checkpoint).resolve()
        and report["task"] == task
        and report["group"] == group
        and int(report["seed"]) == int(seed)
        and int(report["num_envs"]) == int(num_envs)
        and int(report["summary"]["episodes"]) == int(episodes)
    )
except (OSError, ValueError, KeyError, TypeError):
    valid = False
raise SystemExit(0 if valid else 1)
PY
        then
          report_valid=1
        else
          echo "[appendix] stale evaluation report; rerunning ${report}"
        fi
      fi
      if (( report_valid == 0 )); then
        python scripts/rsl_rl/evaluate_policy.py \
          --task "${play_task}" \
          --checkpoint "${latest_checkpoint}" \
          --episodes "${EVAL_EPISODES}" \
          --num_envs "${EVAL_NUM_ENVS}" \
          --seed "${eval_seed}" \
          --group "appendix_${group}_train${seed}" \
          --output "${report}" \
          --headless
      fi
      eval_reports+=("${report}")
    done
    python scripts/rsl_rl/summarize_policy_evaluations.py \
      "${eval_reports[@]}" \
      --output "${eval_dir}/appendix_${group}_train${seed}_seed_summary.json"

    recovery_reports=()
    for eval_seed in ${EVAL_SEEDS}; do
      recovery_report="${eval_dir}/appendix_${group}_train${seed}_recovery_eval${eval_seed}.json"
      recovery_valid=0
      if [[ -s "${recovery_report}" ]]; then
        if python - "${recovery_report}" "${latest_checkpoint}" "${play_task}" "appendix_${group}_train${seed}_recovery" "${eval_seed}" "${EVAL_EPISODES}" "${EVAL_NUM_ENVS}" <<'PY'
import json
import sys
from pathlib import Path

report_path, checkpoint, task, group, seed, episodes, num_envs = sys.argv[1:]
try:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    protocol = report.get("recovery_protocol", {})
    valid = (
        Path(report["checkpoint"]).resolve() == Path(checkpoint).resolve()
        and report["task"] == task
        and report["group"] == group
        and int(report["seed"]) == int(seed)
        and int(report["num_envs"]) == int(num_envs)
        and int(report["summary"]["episodes"]) == int(episodes)
        and int(report["summary"]["recovery_episodes"]) > 0
        and protocol.get("perturbation_step") == 100
        and protocol.get("lateral_velocity_kick_mps") == 0.8
    )
except (OSError, ValueError, KeyError, TypeError):
    valid = False
raise SystemExit(0 if valid else 1)
PY
        then
          recovery_valid=1
        else
          echo "[appendix] stale recovery report; rerunning ${recovery_report}"
        fi
      fi
      if (( recovery_valid == 0 )); then
        python scripts/rsl_rl/evaluate_policy.py \
          --task "${play_task}" \
          --checkpoint "${latest_checkpoint}" \
          --episodes "${EVAL_EPISODES}" \
          --num_envs "${EVAL_NUM_ENVS}" \
          --seed "${eval_seed}" \
          --group "appendix_${group}_train${seed}_recovery" \
          --output "${recovery_report}" \
          --recovery_eval \
          --perturbation_step 100 \
          --lateral_velocity_kick 0.8 \
          --recovery_error_threshold 0.35 \
          --recovery_hold_seconds 0.5 \
          --headless
      fi
      recovery_reports+=("${recovery_report}")
    done
    python scripts/rsl_rl/summarize_policy_evaluations.py \
      "${recovery_reports[@]}" \
      --output "${eval_dir}/appendix_${group}_train${seed}_recovery_seed_summary.json"
  done

  shopt -s nullglob
  train_summaries=()
  recovery_summaries=("${eval_dir}/appendix_${group}_train"*_recovery_seed_summary.json)
  # The ordinary glob also matches *_recovery_seed_summary.json. Filter those
  # files before passing summaries to the non-recovery matrix aggregator.
  for summary in "${eval_dir}/appendix_${group}_train"*_seed_summary.json; do
    [[ -e "${summary}" ]] || continue
    [[ "${summary}" == *_recovery_seed_summary.json ]] && continue
    train_summaries+=("${summary}")
  done
  if (( ${#train_summaries[@]} > 0 )); then
    python scripts/rsl_rl/summarize_appendix_matrix.py \
      "${train_summaries[@]}" \
      --output "${eval_dir}/appendix_${group}_matrix_summary.json"
  fi
  if (( ${#recovery_summaries[@]} > 0 )); then
    python scripts/rsl_rl/summarize_appendix_matrix.py \
      "${recovery_summaries[@]}" \
      --output "${eval_dir}/appendix_${group}_recovery_matrix_summary.json"
  fi
done
