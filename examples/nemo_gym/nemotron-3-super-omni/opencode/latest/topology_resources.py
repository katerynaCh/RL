# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import subprocess
from uuid import UUID


def topology_resources(smi_output: str, rank: int) -> dict[str, int]:
    values = re.findall(r"^\s*ClusterUUID\s*:\s*(\S+)", smi_output, re.MULTILINE)
    if len(values) != 4:
        raise ValueError(f"Expected fabric UUIDs for four GPUs, found {len(values)}")
    domains = {UUID(value) for value in values}
    if len(domains) != 1 or UUID(int=0) in domains:
        raise ValueError(f"Expected one valid NVLink fabric UUID, found {domains}")
    if rank < 0:
        raise ValueError("SLURM_PROCID must be nonnegative")
    domain = str(next(iter(domains)))
    return {
        "worker_units": 4,
        "slurm_managed_ray_cluster": 1,
        f"nvlink_domain_{domain}": 1,
        "topo_rank": rank + 1,
    }


if __name__ == "__main__":
    result = subprocess.run(
        ["nvidia-smi", "-q"], capture_output=True, text=True, check=True, timeout=30
    )
    print(
        json.dumps(topology_resources(result.stdout, int(os.environ["SLURM_PROCID"])))
    )
