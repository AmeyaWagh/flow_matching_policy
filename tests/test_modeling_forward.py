"""Tests for FlowMatchingPolicy.forward() / FlowMatchingModel.compute_loss()."""

import torch

from lerobot_policy_flow_matching import FlowMatchingPolicy


def test_forward_returns_finite_scalar_loss(policy: FlowMatchingPolicy, train_batch: dict[str, torch.Tensor]):
    """forward() on a synthetic training batch returns a finite scalar loss and no output dict."""
    loss, output_dict = policy.forward(train_batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert output_dict is None


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
