# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Opt-in, conversation-independent image tiling for Nemotron Omni rollouts.

The image content, dimensions and model configuration determine its geometry.
Text growth and the addition of another image cannot resize existing images.
This is intentionally a capacity tradeoff: reject an overlength conversation
instead of fitting it by changing the pixels underlying its exact token history.
"""

from typing import Any


def stable_compute_params(
    tiler: Any, media_list: list, num_tokens_available: int
) -> list:
    """Use the original single-image decision procedure with a fixed budget.

    ``process_media`` computes the actual grid from image dimensions. We never
    infer a grid from a token count or reuse a grid for a different image.
    The dynamic request budget is deliberately unused. Keep this signature
    compatible with vLLM's DynamicResolutionImageTiler.compute_params.
    """
    budget = tiler.max_num_tokens_available(text_prompt_length=0)
    budget *= (4 if tiler.PIXEL_SHUFFLE else 1) * (4 if tiler.CONV_MERGING else 1)
    budget = int(max(min(budget, tiler._max_num_patches), tiler._min_num_patches))
    return [tiler.process_media(image, budget)[0] for image in media_list]


def install_stable_image_preprocessing(hf_config: Any) -> None:
    """Install before preprocessing requests in a dedicated generation worker.

    vLLM 0.29.0 provides no per-image geometry override. Patch only its image
    tiler's decision method; normalization, interpolation, feature generation,
    placeholder construction and video processing remain vLLM's implementations.
    Repeated installation is harmless. No mutable per-request geometry or pixel
    cache is retained, so concurrent requests and replica changes are safe.
    """
    if getattr(hf_config, "model_type", None) != "nemotron_h_omni":
        raise ValueError(
            "stable_image_preprocessing currently supports nemotron_h_omni only"
        )
    from vllm.transformers_utils.processors.nano_nemotron_vl import (
        DynamicResolutionImageTiler,
    )

    DynamicResolutionImageTiler.compute_params = stable_compute_params
