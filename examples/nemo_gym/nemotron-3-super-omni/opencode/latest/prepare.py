# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
from dataclasses import fields
from pathlib import Path

import ray
import torch
from nemo_gym import PARENT_DIR
from nemo_gym.cli.setup_command import setup_env_command
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from vllm.engine.arg_utils import AsyncEngineArgs

from nemo_rl.algorithms.single_controller_utils import MasterConfig
from nemo_rl.algorithms.single_controller_utils.config import (
    validate_single_controller_config,
)
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.utils import setup_response_data
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

assert ray.__version__ == "2.58.0", ray.__version__
assert torch.__version__.startswith("2.13."), torch.__version__
vllm = Path(
    "/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
)
actor_root = Path(os.environ["NEMO_RL_VENV_DIR"])
actor_root.mkdir(parents=True, exist_ok=True)
for source in Path("/opt/ray_venvs").iterdir():
    if source.is_dir() and (source / "bin/python").is_file():
        target = actor_root / source.name
        if not target.exists():
            target.symlink_to(source, target_is_directory=True)
for name in (
    "nemo_rl.algorithms.single_controller.SingleControllerActor",
    "nemo_rl.experience.rollout_reassembler_actor.RolloutReassemblerActor",
):
    target = actor_root / name
    if not target.exists():
        target.symlink_to(vllm, target_is_directory=True)
gym_root = Path(os.environ["NEMO_GYM_VENV_DIR"])
for component in (
    "responses_api_models/vllm_model",
    "responses_api_agents/opencode_agent",
    "responses_api_agents/opencode_multimodal_agent",
    "resources_servers/string_match",
    "resources_servers/math_with_judge",
    "resources_servers/mcqa",
    "environment_servers/legacy_agent",
):
    venv = gym_root / component / ".venv"
    venv.parent.mkdir(parents=True, exist_ok=True)
    if not venv.exists():
        venv.symlink_to(vllm, target_is_directory=True)
    command = setup_env_command(
        PARENT_DIR / component,
        OmegaConf.create(
            {
                "head_server_deps": [],
                "python_version": "3.13.14",
                "uv_venv_dir": str(gym_root),
                "skip_venv_if_present": True,
            }
        ),
        "preflight",
    )
    assert "uv pip" not in command and "uv venv" not in command, command
register_omegaconf_resolvers()
config = load_config(os.environ["TRAIN_CONFIG"])
resolved = OmegaConf.to_container(config, resolve=True)
validate_single_controller_config(MasterConfig(**resolved))

unknown = set(config.policy.generation.vllm_kwargs) - {
    field.name for field in fields(AsyncEngineArgs)
}
assert not unknown, f"Unsupported vLLM engine options: {sorted(unknown)}"
assert config.policy.megatron_cfg.tensor_model_parallel_size == 8
assert config.grpo.max_num_steps == 200
assert config.grpo.num_prompts_per_step == 32
assert config.grpo.num_generations_per_prompt == 16
assert config.policy.train_global_batch_size == 512
assert config.cluster.num_nodes == 16
assert config.policy.generation.colocated.resources.num_nodes == 8
assert config.policy.megatron_cfg.expert_model_parallel_size == 16
assert config.policy.megatron_cfg.context_parallel_size == 2
assert config.token_capture.enabled
print("PASS training config and isolated runtime", flush=True)

if "opencode_multimodal_agent" in config.env.nemo_gym:
    assert config.policy.get("is_vlm"), (
        "Single-controller media capture requires policy.is_vlm=true"
    )
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.data.multimodal_utils import uses_image_placeholder

    processor = get_tokenizer(resolved["policy"]["tokenizer"], get_processor=True)
    assert uses_image_placeholder(processor), type(processor).__name__
    print("PASS native media capture processor", type(processor).__name__, flush=True)

policy_nodes = (
    config.cluster.num_nodes - config.policy.generation.colocated.resources.num_nodes
)
world_size = policy_nodes * config.cluster.gpus_per_node
parallel = config.policy.megatron_cfg
dense_group = (
    parallel.tensor_model_parallel_size
    * parallel.pipeline_model_parallel_size
    * parallel.context_parallel_size
)
expert_group = (
    parallel.expert_tensor_parallel_size
    * parallel.expert_model_parallel_size
    * parallel.pipeline_model_parallel_size
)
assert world_size % dense_group == world_size % expert_group == 0
if config.cluster.segment_size is not None:
    assert policy_nodes % config.cluster.segment_size == 0
assert config.policy.train_global_batch_size % (world_size // dense_group) == 0
print(
    "PASS parallel layout",
    policy_nodes,
    "policy nodes",
    "DP",
    world_size // dense_group,
    flush=True,
)

# Schema validation alone cannot catch a loader inherited from an unrelated recipe.
# Exercise the same wrapping, processing and worker process as the controller before
# starting any GPU actors. The last short batch is included to cover every row.
for split in ("train", "validation"):
    effective = {**resolved["data"]["default"], **resolved["data"][split]}
    assert effective["dataset_name"] == "NemoGymDataset", effective
if os.environ.get("SLURM_PROCID", "0") == "0":
    train_dataset, _ = setup_response_data(
        processor, copy.deepcopy(resolved["data"]), env_configs=None, is_vlm=True
    )
    manifest_path = Path(os.environ["TRAIN_CONFIG"]).parent / "dataset-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_rows = manifest["rows"]
    assert len(train_dataset) == expected_rows, len(train_dataset)
    loader = StatefulDataLoader(
        train_dataset,
        batch_size=config.grpo.num_prompts_per_step,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers,
        collate_fn=rl_collate_fn,
        drop_last=False,
    )
    rows_checked = 0
    with Path(config.data.train.data_path).open() as source:
        for batch in loader:
            for row in batch["extra_env_info"]:
                assert row == json.loads(next(source)), rows_checked
                assert row["agent_ref"]["name"] == "opencode_multimodal_agent"
                rows_checked += 1
        assert next(source, None) is None
    assert rows_checked == expected_rows, rows_checked
    print(
        "PASS actual RL dataloader: all",
        rows_checked,
        "Gym rows in order",
        flush=True,
    )
