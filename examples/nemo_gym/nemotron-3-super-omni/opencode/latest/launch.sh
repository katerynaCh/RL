#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export RL_ROOT
RL_ROOT=$(git -C "$HERE" rev-parse --show-toplevel)
if [[ $# -gt 1 || ( $# -eq 1 && $1 != --dry-run ) ]]; then
  echo "Usage: $0 [--dry-run] (see README.md for required environment variables)" >&2
  exit 2
fi
for variable in CONTAINER_IMAGE MODEL_PATH TRAIN_DATA IMAGE_ROOT OPENCODE_CACHE \
  OPENCODE_BIN OPENCODE_SANDBOX_BWRAP_ROOT OPENCODE_TOOL_PYTHON \
  OPENCODE_PLUGIN_DEPS SLURM_ACCOUNT WANDB_PROJECT WANDB_ENTITY; do
  [[ -n ${!variable:-} ]] || { echo "Set $variable; see README.md" >&2; exit 1; }
  export "$variable"
done
export GYM_ROOT=${GYM_ROOT:-$RL_ROOT/3rdparty/Gym-workspace/Gym}
export EXP_NAME=${EXP_NAME:-latest-run}
[[ $EXP_NAME =~ ^[a-zA-Z0-9_.-]+$ ]] || { echo "Invalid EXP_NAME" >&2; exit 1; }
export MM_TRAINER_WANDB_NAME=$EXP_NAME
export MM_TRAINER_WANDB_ID=${MM_TRAINER_WANDB_ID:-$EXP_NAME}
export FULL_WORK_ROOT=${FULL_WORK_ROOT:-$RL_ROOT/results/$EXP_NAME}
export FULL_LOG_ROOT=${FULL_LOG_ROOT:-$RL_ROOT/logs/$EXP_NAME}
export SUPER_MEGATRON_CACHE=${SUPER_MEGATRON_CACHE:-/cache/megatron-$EXP_NAME-tp8-ep16-cp2}
export NCCL_NET_PLUGIN=none NCCL_NET=IB
export WANDB_MODE=${WANDB_MODE:-online}
export UV_BIN
UV_BIN=$(command -v uv)
source "$HERE/sandbox_env.sh"

for path in "$CONTAINER_IMAGE" "$MODEL_PATH/config.json" \
  "$MODEL_PATH/model.safetensors.index.json" "$MODEL_PATH/chat_template.jinja" \
  "$OPENCODE_BIN" "$OPENCODE_CACHE/runtime-overlay/ray/__init__.py" "$TRAIN_DATA" \
  "$GYM_ROOT/responses_api_agents/opencode_multimodal_agent/combined.py" \
  "$GYM_ROOT/resources_servers/string_match/app_passrate.py"; do
  [[ -f $path ]] || { echo "Missing required runtime artifact: $path" >&2; exit 1; }
done
for path in "$IMAGE_ROOT" "$OPENCODE_CACHE/huggingface" "$OPENCODE_CACHE/hf_modules"; do
  [[ -d $path ]] || { echo "Missing runtime directory: $path" >&2; exit 1; }
done
if [[ $WANDB_MODE == online && -z ${WANDB_API_KEY:-} ]]; then
  echo "Export WANDB_API_KEY or select WANDB_MODE=offline." >&2
  exit 1
fi

submit=(sbatch --parsable --account="$SLURM_ACCOUNT" --partition="${SLURM_PARTITION:-batch}"
  --segment=8 --job-name="$EXP_NAME" --output="$FULL_LOG_ROOT/slurm/%x-%j.log")
if [[ ${1:-} == --dry-run ]]; then
  printf '%q ' "${submit[@]}" "$HERE/job.sbatch"
  printf '\n'
  exit 0
fi

active=$(squeue --noheader --user="$USER" --name="$EXP_NAME" --format='%i')
dependency_args=()
if [[ -n ${CONTINUE_AFTER_JOB_ID:-} ]]; then
  [[ $CONTINUE_AFTER_JOB_ID =~ ^[0-9]+$ ]] || { echo "Invalid continuation job ID" >&2; exit 1; }
  parent_name=$(squeue --noheader --jobs="$CONTINUE_AFTER_JOB_ID" --format='%j')
  [[ $parent_name == "$EXP_NAME" ]] || { echo "Continuation parent must belong to this experiment" >&2; exit 1; }
  while IFS= read -r job_id; do
    if [[ -n $job_id && $job_id != "$CONTINUE_AFTER_JOB_ID" ]]; then
      echo "Another job already owns this experiment: $job_id" >&2
      exit 1
    fi
  done <<< "$active"
  export REQUIRE_CHECKPOINT=1
  dependency_args=(--dependency="afterany:$CONTINUE_AFTER_JOB_ID" --kill-on-invalid-dep=yes)
else
  [[ -z $active ]] || { echo "This experiment already has active job(s): $active" >&2; exit 1; }
  export REQUIRE_CHECKPOINT=${REQUIRE_CHECKPOINT:-0}
fi
continuation_count=${CONTINUATION_COUNT:-1}
[[ $continuation_count =~ ^[1-9][0-9]*$ ]] || { echo "CONTINUATION_COUNT must be positive" >&2; exit 1; }
mkdir -p "$FULL_WORK_ROOT" "$FULL_LOG_ROOT/slurm"
for ((index=0; index<continuation_count; index++)); do
  submission=$("${submit[@]}" "${dependency_args[@]}" "$HERE/job.sbatch")
  job_id=${submission%%;*}
  [[ $job_id =~ ^[0-9]+$ ]] || { echo "Unrecognized sbatch response: $submission" >&2; exit 1; }
  printf 'Submitted batch job %s\n' "$job_id"
  dependency_args=(--dependency="afterany:$job_id" --kill-on-invalid-dep=yes)
  export REQUIRE_CHECKPOINT=1
done
