"""Tests for the fixed-step ODE solvers in `ode_solvers.py`."""

import pytest
import torch

from lerobot_policy_flow_matching.ode_solvers import (
    ODE_SOLVERS,
    euler_solve,
    heun_solve,
    integrate,
    rk4_solve,
)


@pytest.mark.parametrize("solve_fn", [euler_solve, heun_solve, rk4_solve])
def test_solvers_are_exact_for_a_constant_velocity_field(solve_fn):
    """A constant `v(x, t) = c` integrates to `noise - c` exactly, regardless of solver order or step count.

    This isolates solver correctness from the learned velocity field: `x(0) = x(1) + integral(v dt from 1
    to 0) = noise - c` for any fixed-step solver, since a constant field has zero curvature (all Taylor
    terms beyond first order vanish).
    """
    noise = torch.randn(4, 8, 3)
    velocity = torch.randn(1, 1, 3)  # broadcasts against noise's trailing dim

    def denoise_fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return velocity.expand_as(x_t)

    for num_steps in (1, 5, 10):
        result = solve_fn(denoise_fn, noise, num_steps)
        expected = noise - velocity
        torch.testing.assert_close(result, expected)


def test_solvers_agree_on_a_linear_time_varying_field():
    """For `v(x, t) = t` (independent of x), all three solvers should closely match the analytic result.

    `dx/dt = t` from `t=1` to `t=0` gives `x(0) = x(1) - integral(t dt, 1, 0) = noise - 0.5`. Euler is only
    first-order accurate so it needs more steps to get close; Heun/RK4 should already be tight at few steps.
    """
    noise = torch.zeros(2, 4, 2)

    def denoise_fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return t.view(-1, *([1] * (x_t.ndim - 1))).expand_as(x_t)

    expected = noise - 0.5

    euler_result = euler_solve(denoise_fn, noise, num_steps=50)
    torch.testing.assert_close(euler_result, expected, atol=1e-2, rtol=0)

    for solve_fn in (heun_solve, rk4_solve):
        result = solve_fn(denoise_fn, noise, num_steps=5)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=0)


def test_call_counts_match_solver_order():
    """heun calls denoise_fn 2x per step, rk4 4x per step, euler 1x per step."""
    noise = torch.zeros(1, 1, 1)
    num_steps = 3
    call_counts = {"euler": 1, "heun": 2, "rk4": 4}

    for name, expected_calls_per_step in call_counts.items():
        calls = 0

        def denoise_fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            nonlocal calls
            calls += 1
            return torch.zeros_like(x_t)

        integrate(name, denoise_fn, noise, num_steps)
        assert calls == expected_calls_per_step * num_steps


def test_integrate_dispatches_by_name():
    """integrate() with a solver name matches calling that solver's function directly."""
    noise = torch.randn(2, 3, 2)

    def denoise_fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x_t * 0.1

    for name, solve_fn in ODE_SOLVERS.items():
        torch.testing.assert_close(
            integrate(name, denoise_fn, noise, num_steps=4),
            solve_fn(denoise_fn, noise, num_steps=4),
        )


def test_integrate_rejects_unknown_solver():
    """integrate() raises ValueError for a solver name not in ODE_SOLVERS."""
    with pytest.raises(ValueError, match="Unknown `ode_solver`"):
        integrate("not_a_real_solver", lambda x, t: x, torch.zeros(1, 1, 1), num_steps=1)
