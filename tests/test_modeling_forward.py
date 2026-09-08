"""Tests for FlowMatchingPolicy.forward() / FlowMatchingModel.compute_loss()."""

import pytest
import torch

from conftest import ACTION_DIM, IMAGE_SHAPE, STATE_DIM, make_config
from lerobot_policy_flow_matching import FlowMatchingConfig, FlowMatchingPolicy


def test_forward_returns_finite_scalar_loss(policy: FlowMatchingPolicy, train_batch: dict[str, torch.Tensor]):
    """forward() on a synthetic training batch returns a finite scalar loss and a metrics dict."""
    loss, output_dict = policy.forward(train_batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert isinstance(output_dict, dict)


def test_forward_loss_requires_grad_in_train_mode(
    policy: FlowMatchingPolicy, train_batch: dict[str, torch.Tensor]
):
    """The loss is connected to the policy's parameters (i.e. backward() would update them)."""
    policy.train()
    loss, _ = policy.forward(train_batch)
    loss.backward()
    grads = [p.grad for p in policy.get_optim_params() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)


def test_do_mask_loss_for_padding(policy: FlowMatchingPolicy, train_batch: dict[str, torch.Tensor]):
    """With do_mask_loss_for_padding enabled, fully-padded actions are excluded from the loss."""
    policy.config.do_mask_loss_for_padding = True
    batch = dict(train_batch)
    batch["action_is_pad"] = torch.ones_like(batch["action_is_pad"])
    loss, _ = policy.forward(batch)
    # All timesteps masked out -> falls back to the num_valid.clamp_min(1) branch, still finite.
    assert torch.isfinite(loss)


def _batch_for(config: FlowMatchingConfig, batch_size: int = 4) -> dict[str, torch.Tensor]:
    """A synthetic training batch sized for the given config (mirrors the `train_batch` fixture)."""
    return {
        "observation.state": torch.randn(batch_size, config.n_obs_steps, STATE_DIM),
        "observation.image": torch.rand(batch_size, config.n_obs_steps, *IMAGE_SHAPE),
        "action": torch.randn(batch_size, config.horizon, ACTION_DIM),
        "action_is_pad": torch.zeros(batch_size, config.horizon, dtype=torch.bool),
    }


def test_ground_truth_trajectory_metrics_present_every_call():
    """gt_path_length/gt_path_smoothness are always in output_dict, and are non-negative and finite."""
    config = make_config()
    policy = FlowMatchingPolicy(config)
    policy.eval()  # ground-truth metrics don't depend on train/eval mode
    for _ in range(3):
        _, output_dict = policy.forward(_batch_for(config))
        assert output_dict["gt_path_length"] >= 0
        assert output_dict["gt_path_smoothness"] >= 0


def test_predicted_trajectory_metrics_gated_by_log_freq():
    """pred_* metrics appear only every trajectory_metrics_log_freq training calls, starting at call 0."""
    config = make_config(trajectory_metrics_log_freq=3)
    policy = FlowMatchingPolicy(config)
    policy.train()
    batch = _batch_for(config)

    presence = []
    for _ in range(7):
        _, output_dict = policy.forward(dict(batch))
        presence.append("pred_path_length" in output_dict)

    assert presence == [True, False, False, True, False, False, True]


def test_predicted_trajectory_metrics_skipped_in_eval_mode():
    """pred_* metrics are never computed outside of training (e.g. the periodic offline eval-loss pass)."""
    config = make_config(trajectory_metrics_log_freq=1)  # would fire on every call, if not gated by eval mode
    policy = FlowMatchingPolicy(config)
    policy.eval()
    batch = _batch_for(config)
    for _ in range(3):
        _, output_dict = policy.forward(dict(batch))
        assert "pred_path_length" not in output_dict
        assert "pred_path_smoothness" not in output_dict


def test_log_trajectory_metrics_disabled():
    """With log_trajectory_metrics=False, output_dict is empty regardless of mode or step."""
    config = make_config(log_trajectory_metrics=False, trajectory_metrics_log_freq=1)
    policy = FlowMatchingPolicy(config)
    policy.train()
    _, output_dict = policy.forward(_batch_for(config))
    assert output_dict == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_trajectory_metrics_work_on_cuda():
    """Regression test: robometric_frame metrics default to CPU and must be moved to the batch's device
    before use, or `.update()` raises on any non-CPU batch (caught via a real GPU training run, not by
    the CPU-only tests above).
    """
    config = make_config(device="cuda", trajectory_metrics_log_freq=1)
    policy = FlowMatchingPolicy(config).to("cuda")
    policy.train()
    batch = {k: v.to("cuda") for k, v in _batch_for(config).items()}
    loss, output_dict = policy.forward(batch)
    assert torch.isfinite(loss)
    assert "pred_path_length" in output_dict
