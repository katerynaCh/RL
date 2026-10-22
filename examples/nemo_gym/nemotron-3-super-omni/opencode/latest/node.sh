#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
RANK=$SLURM_PROCID
export LAUNCH_HOME=$RUN/harness
export TRAIN_CONFIG=$LAUNCH_HOME/config.yaml
BRIDGE=/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
POLICY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
DRIVER=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python
export PATH=/stock:/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH=/cache/runtime-overlay:/tmp/opencode-mcore:/cache/hf_modules:/opt/nemo-rl:/opt/nemo-rl/3rdparty/Gym-workspace/Gym:$BRIDGE/src:$BRIDGE/3rdparty/Megatron-LM
export HF_HOME=/cache/huggingface HF_MODULES_CACHE=/cache/hf_modules
export XDG_CACHE_HOME=$RUN/cache/node-$RANK UV_CACHE_DIR=$RUN/cache/uv
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export NEMO_RL_VENV_DIR=/tmp/opencode-ray-venvs NRL_FORCE_REBUILD_VENVS=false
export NEMO_GYM_VENV_DIR=/tmp/opencode-gym-venvs
export XDG_CONFIG_HOME=$RUN/config/node-$RANK OPENCODE_DISABLE_MODELS_FETCH=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 RAY_USAGE_STATS_ENABLED=0 RAY_DEDUP_LOGS=0
export CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_NVLS_ENABLE=0 VLLM_TRITON_FORCE_FIRST_CONFIG=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False TORCH_CUDA_ARCH_LIST=10.0
export MM_TRAINER_MODEL_PATH=/model MM_TRAINER_RESULTS_DIR=/work
export SUPER_MEGATRON_CACHE=$SUPER_MEGATRON_CACHE
export NRL_MEGATRON_CHECKPOINT_DIR=$SUPER_MEGATRON_CACHE
export MM_TRAINER_WANDB_ID=$MM_TRAINER_WANDB_ID
export WANDB_MODE=${WANDB_MODE:-disabled}
export MEGATRON_CONFIG_LOCK_DIR=/cache/model-locks
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1
export RAY_num_grpc_internal_threads=1 RAY_worker_num_grpc_internal_threads=1
export RAY_num_server_call_thread=1 RAY_gcs_server_rpc_server_thread_num=1 RAY_gcs_server_rpc_client_thread_num=1
# Check the checkpoint with the image's interpreter, before starting any workers.
# All ranks see the same published checkpoint and exit together for a finished run.
if [[ ${REQUIRE_CHECKPOINT:-0} == 1 ]]; then
 decision=$("$DRIVER" -I "$LAUNCH_HOME/continuation_guard.py" "$MM_TRAINER_RESULTS_DIR/checkpoints")
 echo "$decision"
 if [[ $decision == complete* ]]; then exit 0; fi
fi
mkdir -p "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME" "$MEGATRON_CONFIG_LOCK_DIR" /tmp/opencode-mcore
# Compile the pinned source's dataset helper in a disposable copy. No source edits.
cp -a "$BRIDGE/3rdparty/Megatron-LM/megatron" /tmp/opencode-mcore/
LIBEXT=$(uv run --no-project --python "$POLICY" "$POLICY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
CPPFLAGS=$(uv run --no-project --python "$POLICY" "$POLICY" -m pybind11 --includes)
export MAKEFLAGS="LIBEXT=$LIBEXT"
timeout -k 5s 120s make -C /tmp/opencode-mcore/megatron/core/datasets "CPPFLAGS=$CPPFLAGS"
uv run --no-project --python "$DRIVER" "$DRIVER" "$LAUNCH_HOME/prepare.py"
cleanup() {
 code=$?
 if (( code != 0 )); then touch "$RUN/FAILED-$RANK"; fi
 mkdir -p "$RUN/logs/ray-$RANK"
 cp -a /tmp/ray/session_latest/logs/. "$RUN/logs/ray-$RANK/" 2>/dev/null || true
 timeout -k 3s 15s "$DRIVER" -m ray.scripts.scripts stop --force || true
}
trap cleanup EXIT
export RAY_ADDRESS=$HEAD_IP:6379
RAY_RESOURCES='{"worker_units":4,"slurm_managed_ray_cluster":1}'
if [[ ${SLURM_JOB_SEGMENT_SIZE:-0} != 0 ]]; then
 RAY_RESOURCES=$("$DRIVER" "$RUN/harness/topology_resources.py")
 echo "TOPOLOGY RESOURCES $SLURMD_NODENAME $RAY_RESOURCES"
fi
if [[ $RANK == 0 ]]; then
 timeout -k 5s 120s "$DRIVER" -m ray.scripts.scripts start --head --node-ip-address="$HEAD_IP" --port=6379 \
  --num-gpus=4 --resources="$RAY_RESOURCES" --include-dashboard=false --disable-usage-stats
 touch "$RUN/HEAD_READY"
 timeout -k 5s 240s uv run --no-project --python "$DRIVER" "$DRIVER" "$LAUNCH_HOME/wait_cluster.py"
 timeout -k 15s "${DRIVER_TIMEOUT_SECONDS:-3000}s" uv run --no-project --python "$DRIVER" "$DRIVER" examples/run_grpo_single_controller.py \
  --config "$TRAIN_CONFIG" > "$RUN/logs/driver.log" 2>&1
 touch "$RUN/DRIVER_COMPLETED"
else
 deadline=$((SECONDS + 240))
 until [[ -f "$RUN/HEAD_READY" ]]; do
  (( SECONDS < deadline )) || exit 1
  sleep 2
 done
 timeout -k 5s 120s "$DRIVER" -m ray.scripts.scripts start --address="$RAY_ADDRESS" --num-gpus=4 \
  --resources="$RAY_RESOURCES" --disable-usage-stats
 deadline=$((SECONDS + ${WORKER_TIMEOUT_SECONDS:-3200}))
 until [[ -f "$RUN/DRIVER_COMPLETED" ]]; do
  (( SECONDS < deadline )) || exit 1
  if compgen -G "$RUN/FAILED-*" >/dev/null; then exit 1; fi
  sleep 5
 done
fi
