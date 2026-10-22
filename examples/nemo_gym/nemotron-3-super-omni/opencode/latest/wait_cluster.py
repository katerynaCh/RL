# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time

import ray

ray.init(address="auto")
deadline = time.monotonic() + 200
while time.monotonic() < deadline:
    nodes = [node for node in ray.nodes() if node["Alive"]]
    if len(nodes) == int(os.environ["SLURM_JOB_NUM_NODES"]):
        assert ray.cluster_resources()["GPU"] == 4 * len(nodes)
        segment_size = int(os.environ.get("SLURM_JOB_SEGMENT_SIZE", "0"))
        if segment_size:
            from nemo_rl.distributed.virtual_cluster import (
                NVLINK_DOMAIN_UNKNOWN,
                get_ray_cluster_topology,
                select_segment_nodes,
            )

            topology = get_ray_cluster_topology()
            assert all(
                domain != NVLINK_DOMAIN_UNKNOWN for domain, _ in topology.values()
            )
            selected, remaining = select_segment_nodes(
                topology, segment_size, segment_size
            )
            print(
                "PASS topology placement", selected, "remaining", remaining, flush=True
            )
        print("CLUSTER READY", len(nodes), ray.cluster_resources(), flush=True)
        break
    time.sleep(3)
else:
    raise TimeoutError("Full Ray allocation did not register")
