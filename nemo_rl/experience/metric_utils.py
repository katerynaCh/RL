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

"""Shared aggregation helpers for rollout metrics."""

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def is_histogram_metric(name: str) -> bool:
    """Return whether a metric key represents raw histogram observations."""
    return name.startswith("histogram/") or name.endswith("/histogram")


def calculate_single_metric(
    values: Sequence[float | int], batch_size: int, key_name: str
) -> dict:
    """Compute summary statistics for a metric as slash-prefixed keys.

    Args:
        values: Per-sample metric values to aggregate.
        batch_size: Denominator for the mean (sum(values) / batch_size, not len(values)); stddev still uses len(values).
        key_name: Prefix for the returned metric keys (e.g. "total_reward").

    Returns:
        Dict mapping "{key_name}/{stat}" to its value for stat in mean, max, min,
        median, stddev (nan for a single value), and histogram. Histogram values
        remain backend-agnostic raw observations until the logger serializes them.
    """
    return {
        f"{key_name}/mean": sum(values) / batch_size,
        f"{key_name}/max": max(values),
        f"{key_name}/min": min(values),
        f"{key_name}/median": statistics.median(values),
        f"{key_name}/stddev": statistics.stdev(values) if len(values) > 1 else math.nan,
        f"{key_name}/histogram": list(values),
    }


def pct(values: Sequence[float | int], p: float) -> float:
    """Percentile helper for buffer starvation diagnostics."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = min(int(len(sorted_v) * p / 100), len(sorted_v) - 1)
    return float(sorted_v[idx])


@dataclass(frozen=True)
class RewardBreakdown:
    """Logging-only answer score and tool shaping; never estimator inputs."""

    base_reward: float
    auxiliary_reward: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.base_reward, self.auxiliary_reward)):
            raise ValueError("Reward breakdown values must be finite")


def extract_reward_breakdown(result: Mapping[str, Any]) -> RewardBreakdown | None:
    """Read the OpenCode agent's existing result without changing its reward."""
    tools = result.get("opencode_tool_reward")
    if tools is None:
        return None
    breakdown = RewardBreakdown(
        base_reward=float(tools["base_reward"]),
        auxiliary_reward=float(tools["auxiliary_reward"]),
    )
    if not math.isclose(
        breakdown.base_reward + breakdown.auxiliary_reward,
        float(result["reward"]),
        rel_tol=1e-5,
        abs_tol=1e-6,
    ):
        raise ValueError("OpenCode base and auxiliary rewards must sum to reward")
    return breakdown
