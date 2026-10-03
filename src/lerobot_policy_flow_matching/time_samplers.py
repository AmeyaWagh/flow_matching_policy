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
"""Training-time flow-matching timestep samplers.

`FlowMatchingModel.compute_loss` draws a random `t` per training example at which the model is trained to
predict the (path-constant) velocity `noise - actions`. This module factors that draw out behind a small,
distribution-name based dispatcher -- mirroring `ode_solvers.py`'s solver dispatch -- so alternative
time-sampling distributions can be swapped in via `FlowMatchingConfig.time_sampling_distribution` without
touching `modeling_flow_matching.py`.

Recall this policy's convention: `t=0` is the clean action trajectory (the target), `t=1` is pure noise.
"beta" (openpi/pi0's convention, via `lerobot.policies.common.flow_matching.sample_time_beta`) samples
close to uniformly across `t`. "logit_normal" instead concentrates training mass around a tunable point in
`(0, 1)` -- e.g. near `t=0` (the target), with a negative `time_sampling_logit_normal_mean` -- so the model
gets denser supervision for whichever regime matters most, at the cost of sparser supervision elsewhere.
The motivating hypothesis (see `docs/comparison_pusht.md`): PushT rollouts fail almost entirely by landing
just *below* the success threshold rather than diverging outright, which points at insufficient precision
in the final, small-`t` refinement steps of ODE integration -- exactly the regime a `t=0`-shifted
logit-normal would oversample during training.
"""

from typing import TYPE_CHECKING

import torch
from lerobot.policies.common.flow_matching import sample_time_beta
from torch import Tensor

if TYPE_CHECKING:
    from .configuration_flow_matching import FlowMatchingConfig

TIME_SAMPLING_DISTRIBUTIONS = ("beta", "logit_normal")


def sample_time_logit_normal(bsize: int, device: torch.device, *, mean: float, std: float) -> Tensor:
    """Logit-normal timesteps: `t = sigmoid(z)`, `z ~ N(mean, std)`.

    A negative `mean` concentrates sampled `t` toward 0 (the clean action trajectory); a positive `mean`
    concentrates toward 1 (noise); `mean=0` concentrates around `t=0.5`. This is the "SD3-style" shifted
    time-sampling trick, adapted here to bias toward whichever end of `(0, 1)` matters most for a given
    experiment rather than SD3's original mid-range concentration.
    """
    z = torch.normal(mean=mean, std=std, size=(bsize,), dtype=torch.float32, device=device)
    return torch.sigmoid(z)


def sample_time(distribution: str, bsize: int, device: torch.device, config: "FlowMatchingConfig") -> Tensor:
    """Dispatch to the named time-sampling distribution in `TIME_SAMPLING_DISTRIBUTIONS`.

    Each distribution reads its own parameters off `config` (they don't share a signature, unlike
    `ode_solvers.ODE_SOLVERS`, since e.g. "beta" needs alpha/beta/scale/offset while "logit_normal" needs
    mean/std).
    """
    if distribution == "beta":
        return sample_time_beta(
            bsize,
            device,
            alpha=config.time_sampling_alpha,
            beta=config.time_sampling_beta,
            scale=config.time_sampling_scale,
            offset=config.time_sampling_offset,
        )
    if distribution == "logit_normal":
        return sample_time_logit_normal(
            bsize,
            device,
            mean=config.time_sampling_logit_normal_mean,
            std=config.time_sampling_logit_normal_std,
        )
    raise ValueError(
        f"Unknown `time_sampling_distribution` {distribution!r}. "
        f"Available distributions: {list(TIME_SAMPLING_DISTRIBUTIONS)}."
    )
