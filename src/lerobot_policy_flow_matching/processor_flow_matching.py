#!/usr/bin/env python

# Copyright 2026 Ameya Wagh. All rights reserved.
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
"""Pre/post-processor pipelines for FlowMatchingPolicy."""

from typing import Any

import torch
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, make_default_pre_post_processors

from .configuration_flow_matching import FlowMatchingConfig


def make_flow_matching_pre_post_processors(
    config: FlowMatchingConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Construct the pre/post-processor pipelines for a flow-matching policy.

    Same feature set and `normalization_mapping` as `DiffusionConfig`, so this delegates directly to
    lerobot's default pipeline (rename -> batch -> device -> normalize, and unnormalize -> cpu) rather than
    composing custom steps.

    Args:
        config: The configuration object for the flow-matching policy.
        dataset_stats: A dictionary of statistics used for normalization.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """
    return make_default_pre_post_processors(config, dataset_stats)
