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
"""ODE solvers for sampling a flow-matching policy's learned velocity field.

`FlowMatchingModel.conditional_sample` needs to integrate `dx/dt = v(x, t)` from `t=1` (noise) to `t=0`
(the clean action trajectory) using the UNet's predicted velocity `v`. lerobot's own
`lerobot.policies.common.flow_matching.euler_integrate` only implements forward-Euler (plus an RTC
guidance hook this policy doesn't use); this module factors integration out behind a small, solver-name
based dispatcher so other fixed-step solvers can be swapped in via `FlowMatchingConfig.ode_solver` without
touching `modeling_flow_matching.py`.

All solvers share the same fixed-step schedule as `euler_integrate` (`dt = -1/num_steps`,
`time = 1 + step*dt`) so that `num_inference_steps` means the same thing -- the number of *steps* -- across
solvers. Higher-order solvers call `denoise_fn` more than once per step (2x for `heun`, 4x for `rk4`), so
they cost proportionally more UNet forward passes at equal `num_inference_steps`; that's the tradeoff being
swept when comparing solvers, not something to normalize away.
"""

from collections.abc import Callable

import torch
from torch import Tensor

DenoiseFn = Callable[[Tensor, Tensor], Tensor]


def _time_tensor(time: float, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Broadcast a scalar flow-matching time to the `(batch_size,)` shape `denoise_fn` expects."""
    return torch.tensor(time, dtype=dtype, device=device).expand(batch_size)


def euler_solve(denoise_fn: DenoiseFn, noise: Tensor, num_steps: int) -> Tensor:
    """Forward-Euler (1st-order): `x_{t+dt} = x_t + dt * v(x_t, t)`. 1 `denoise_fn` call per step."""
    bsize = noise.shape[0]
    dt = -1.0 / num_steps
    x_t = noise
    for step in range(num_steps):
        time = 1.0 + step * dt
        v_t = denoise_fn(x_t, _time_tensor(time, bsize, noise.device, noise.dtype))
        x_t = x_t + dt * v_t
    return x_t


def heun_solve(denoise_fn: DenoiseFn, noise: Tensor, num_steps: int) -> Tensor:
    """Heun's method (2nd-order predictor-corrector / improved Euler). 2 `denoise_fn` calls per step.

    Predicts with a forward-Euler step, then corrects using the average of the velocity at the start and
    (predicted) end of the step.
    """
    bsize = noise.shape[0]
    dt = -1.0 / num_steps
    x_t = noise
    for step in range(num_steps):
        t0 = 1.0 + step * dt
        t1 = t0 + dt
        v0 = denoise_fn(x_t, _time_tensor(t0, bsize, noise.device, noise.dtype))
        x_euler = x_t + dt * v0
        v1 = denoise_fn(x_euler, _time_tensor(t1, bsize, noise.device, noise.dtype))
        x_t = x_t + dt * 0.5 * (v0 + v1)
    return x_t


def rk4_solve(denoise_fn: DenoiseFn, noise: Tensor, num_steps: int) -> Tensor:
    """Classical 4th-order Runge-Kutta. 4 `denoise_fn` calls per step."""
    bsize = noise.shape[0]
    dt = -1.0 / num_steps
    x_t = noise
    for step in range(num_steps):
        t0 = 1.0 + step * dt
        t_mid = t0 + dt / 2
        t1 = t0 + dt
        k1 = denoise_fn(x_t, _time_tensor(t0, bsize, noise.device, noise.dtype))
        k2 = denoise_fn(x_t + dt / 2 * k1, _time_tensor(t_mid, bsize, noise.device, noise.dtype))
        k3 = denoise_fn(x_t + dt / 2 * k2, _time_tensor(t_mid, bsize, noise.device, noise.dtype))
        k4 = denoise_fn(x_t + dt * k3, _time_tensor(t1, bsize, noise.device, noise.dtype))
        x_t = x_t + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_t


ODE_SOLVERS: dict[str, Callable[[DenoiseFn, Tensor, int], Tensor]] = {
    "euler": euler_solve,
    "heun": heun_solve,
    "rk4": rk4_solve,
}


def integrate(solver: str, denoise_fn: DenoiseFn, noise: Tensor, num_steps: int) -> Tensor:
    """Dispatch to the named solver in `ODE_SOLVERS` and integrate `noise` from `t=1` to `t=0`."""
    try:
        solve_fn = ODE_SOLVERS[solver]
    except KeyError:
        raise ValueError(
            f"Unknown `ode_solver` {solver!r}. Available solvers: {sorted(ODE_SOLVERS)}."
        ) from None
    return solve_fn(denoise_fn, noise, num_steps)
