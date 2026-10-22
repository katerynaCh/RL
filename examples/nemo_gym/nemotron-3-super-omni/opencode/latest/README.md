# Latest OpenCode training run

This Slurm/Pyxis launcher runs multimodal GRPO with the Gym
`opencode_multimodal_agent`. It uses 16 nodes with four GPUs each: eight policy
nodes and eight generation nodes, placed in eight-node segments.

The recipe uses 32 prompts × 16 generations, 49,152-token context, eight agent
turns, retained reasoning history, and no MTP. It trains for 200 steps, saves
every ten steps, and requests a checkpoint and exit after three hours of a
four-hour allocation. Sampling is temperature 1 / top-p 1. The agent uses the
OpenCode system prompt in `config.yaml`, without an added visual prefix.

Tool penalties apply only to answers receiving full credit. Bash-call penalties
are 0, 0, 0, −0.025, −0.05, −0.075, −0.10; invalid calls receive −0.05 and duplicate
calls −0.02. No positive tool bonus is configured. Groups need at least half
their rollouts valid; failures are not refilled, and each step must retain at
least 90% of its prompt groups. See the recipe for all optimizer and model settings.

## Runtime prerequisites

- Slurm with Pyxis and `--segment` support, on ARM64 hosts with NVLink domains.
- A NeMo-RL container with prebuilt environments under `/opt/ray_venvs`,
  PyTorch 2.13 and compatible vLLM/model support. The driver uses the vLLM
  environment and preflight checks Ray 2.58.0.
- An OpenCode 1.17.11 binary and Bubblewrap sandbox with a Python environment
  containing Pillow, NumPy, SciPy and OpenCV, plus OpenCode plugin dependencies.
- A cache containing `runtime-overlay/`, `huggingface/` and `hf_modules/`.
  `runtime-requirements.txt` lists the overlay versions. Use a separate converted
  Megatron cache for each initial model; do not reuse one for different weights.
- A Gym checkout containing `opencode_multimodal_agent`, its model adapter, and
  the optional `resources_servers/string_match/app_passrate.py` entrypoint.
  Set `GYM_ROOT` explicitly until the repository's Gym submodule is pinned to
  a revision containing those components. The launcher records its actual revision.
- A Gym JSONL dataset with `agent_ref.name: opencode_multimodal_agent` and image
  paths under `/images`. Dataset and initial model weights are supplied by you;
  neither is included here. The same data is used for the disabled validation path.

## Launch

Export the following values in your shell. Credentials are read from the
environment; do not put them into the launcher or recipe.

```bash
export CONTAINER_IMAGE=/absolute/path/to/training.sqsh
export MODEL_PATH=/absolute/path/to/hf-model
export TRAIN_DATA=/absolute/path/to/train.jsonl
export IMAGE_ROOT=/absolute/path/to/images
export GYM_ROOT=/absolute/path/to/Gym
export OPENCODE_CACHE=/absolute/path/to/runtime-cache
export OPENCODE_BIN=/absolute/path/to/opencode
export OPENCODE_SANDBOX_BWRAP_ROOT=/absolute/path/to/bubblewrap-runtime
export OPENCODE_TOOL_PYTHON=/absolute/path/to/tool-python
export OPENCODE_PLUGIN_DEPS=/absolute/path/to/plugin-dependencies
export SLURM_ACCOUNT=your-account
export SLURM_PARTITION=batch
export WANDB_PROJECT=your-project
export WANDB_ENTITY=your-entity
export WANDB_MODE=offline  # Or online, with WANDB_API_KEY already exported.
export EXP_NAME=latest-run

bash examples/nemo_gym/nemotron-3-super-omni/opencode/latest/launch.sh --dry-run
bash examples/nemo_gym/nemotron-3-super-omni/opencode/latest/launch.sh
```

The dry run validates local artifacts and prints the submission command without
contacting Slurm. Outputs go to `results/$EXP_NAME` and `logs/$EXP_NAME`; override
`FULL_WORK_ROOT` and `FULL_LOG_ROOT` to place them elsewhere. Change `EXP_NAME`
for a fresh run. Reusing it allows checkpoint restoration; the default W&B ID
also follows that name, or can be set through `MM_TRAINER_WANDB_ID`.

Set `CONTINUATION_COUNT` to request sequential allocations. To chain after an
already running allocation, set `CONTINUE_AFTER_JOB_ID` to its job ID. Dependent
allocations require a published checkpoint and exit without training once the
target step is reached. The launch guard rejects concurrent jobs for the same
experiment name.
