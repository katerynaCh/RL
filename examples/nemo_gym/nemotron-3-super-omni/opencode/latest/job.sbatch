#!/usr/bin/env bash
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

export RUN=/work/attempts/$SLURM_JOB_ID
export MM_TRAINER_RESULTS_DIR=/work
export DRIVER_TIMEOUT_SECONDS=13900 WORKER_TIMEOUT_SECONDS=14100
HOST_RUN=$FULL_WORK_ROOT/attempts/$SLURM_JOB_ID
mkdir -p "$HOST_RUN/logs" "$HOST_RUN/harness"
git -C "$RL_ROOT" rev-parse HEAD > "$HOST_RUN/pins.txt"
git -C "$RL_ROOT" submodule status --recursive >> "$HOST_RUN/pins.txt"
git -C "$RL_ROOT" status --porcelain=v1 > "$HOST_RUN/source-status.txt"
LAUNCH_HOME=$RL_ROOT/examples/nemo_gym/nemotron-3-super-omni/opencode/latest
cp -a "$LAUNCH_HOME/." "$HOST_RUN/harness/"
# The snapshot is no longer beside the repository's inherited recipe.
sed -i 's|^defaults: .*|defaults: /opt/nemo-rl/examples/configs/recipes/vlm/vlm_grpo-supervl3p5-clevr-24n4g-megatron-tp8ep8-single-controller.v1.yaml|' "$HOST_RUN/harness/config.yaml"
awk 'END {printf "{\"rows\": %d}\n", NR}' "$TRAIN_DATA" > "$HOST_RUN/harness/dataset-manifest.json"
sha256sum "$TRAIN_DATA" > "$HOST_RUN/dataset.sha256"
scontrol show job "$SLURM_JOB_ID" > "$HOST_RUN/slurm-job.txt"
NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE=${NODES%%$'\n'*}
export HEAD_IP
HEAD_IP=$(getent ahostsv4 "$HEAD_NODE" | awk 'NR==1 {print $1}')
test -n "$HEAD_IP"
HF_CACHE=$(realpath "$OPENCODE_CACHE/huggingface")
HF_MODULES=$(realpath "$OPENCODE_CACHE/hf_modules")
MOUNTS="$RL_ROOT:/opt/nemo-rl:ro,$FULL_WORK_ROOT:/work:rw,$FULL_LOG_ROOT:/logs:rw,$OPENCODE_CACHE:/cache:rw,$HF_CACHE:/cache/huggingface:rw,$HF_MODULES:/cache/hf_modules:rw,$MODEL_PATH:/model:ro,$IMAGE_ROOT:/images:ro,$TRAIN_DATA:/data/train.jsonl:ro,$OPENCODE_BIN:/stock/opencode:ro,$UV_BIN:/usr/local/bin/uv:ro${OPENCODE_SANDBOX_MOUNTS:-}"
MOUNTS+=",$GYM_ROOT:/opt/nemo-rl/3rdparty/Gym-workspace/Gym:ro"
git -C "$GYM_ROOT" rev-parse HEAD > "$HOST_RUN/gym-revision.txt"
git -C "$GYM_ROOT" diff > "$HOST_RUN/gym-source.patch"
git -C "$GYM_ROOT" status --porcelain=v1 > "$HOST_RUN/gym-source-status.txt"
timeout -k 20s 14300s srun --kill-on-bad-exit=1 --wait=30 --no-container-mount-home \
  --container-image="$CONTAINER_IMAGE" --container-mounts="$MOUNTS" \
  --container-workdir=/opt/nemo-rl --output="$HOST_RUN/logs/node-%t.log" \
  bash "$RUN/harness/node.sh"
