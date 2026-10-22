# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward component extraction and masked step-level logging."""

from __future__ import annotations

import pytest
import torch

from nemo_rl.algorithms.single_controller_utils.utils import (
    reduce_advantage_pump_metrics,
)
from nemo_rl.experience.metric_utils import RewardBreakdown, extract_reward_breakdown


def test_extracts_existing_opencode_response_without_changing_reward() -> None:
    result = {
        "reward": 0.95,
        "opencode_tool_reward": {"base_reward": 1.0, "auxiliary_reward": -0.05},
    }
    assert extract_reward_breakdown(result) == RewardBreakdown(1.0, -0.05)
    assert result["reward"] == 0.95
    assert extract_reward_breakdown({"reward": 0.5}) is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_components(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        RewardBreakdown(bad, 0.0)


def test_rejects_inconsistent_agent_breakdown() -> None:
    with pytest.raises(ValueError, match="sum"):
        extract_reward_breakdown(
            {
                "reward": 1.0,
                "opencode_tool_reward": {"base_reward": 1.0, "auxiliary_reward": 0.1},
            }
        )


def test_step_means_use_same_weighting_and_masks_across_unequal_chunks() -> None:
    out = reduce_advantage_pump_metrics(
        [torch.tensor([1.1, 0.2]), torch.tensor([99.0, 0.95, 0.0])],
        [],
        [],
        sample_masks=[torch.tensor([1.0, 0.5]), torch.tensor([0.0, 1.0, 0.0])],
        reward_breakdowns=[
            RewardBreakdown(1.0, 0.1),
            RewardBreakdown(0.0, 0.2),
            None,
            RewardBreakdown(1.0, -0.05),
            None,
        ],
    )
    assert out["base_reward"] == pytest.approx(0.8)
    assert out["auxiliary_reward"] == pytest.approx(0.06)
    assert out["reward"] == pytest.approx(out["base_reward"] + out["auxiliary_reward"])
    assert out["reward_breakdown_coverage"] == 1.0


def test_old_checkpoint_rows_suppress_partial_means() -> None:
    out = reduce_advantage_pump_metrics(
        [torch.tensor([1.1, 0.2])],
        [],
        [],
        sample_masks=[torch.tensor([1.0, 1.0])],
        reward_breakdowns=[RewardBreakdown(1.0, 0.1), None],
    )
    assert out["reward_breakdown_coverage"] == 0.5
    assert "base_reward" not in out and "auxiliary_reward" not in out
    assert out["reward"] == pytest.approx(0.65)


def test_zero_valid_rows_do_not_emit_misleading_components() -> None:
    out = reduce_advantage_pump_metrics(
        [torch.tensor([1.1])],
        [],
        [],
        sample_masks=[torch.zeros(1)],
        reward_breakdowns=[None],
    )
    assert out["reward"] == 0.0
    assert out["reward_breakdown_coverage"] == 0.0
    assert "base_reward" not in out


def test_reducer_checks_alignment_and_training_reward_consistency() -> None:
    with pytest.raises(ValueError, match="align"):
        reduce_advantage_pump_metrics([torch.ones(1)], [], [], reward_breakdowns=[])
    with pytest.raises(ValueError, match="sum"):
        reduce_advantage_pump_metrics(
            [torch.ones(1)], [], [], reward_breakdowns=[RewardBreakdown(1.0, 0.1)]
        )
