"""Tests for FlowMatchingPolicy's action-generation paths: predict_action_chunk() and select_action()."""

from unittest.mock import patch

import torch

from conftest import ACTION_DIM, BATCH_SIZE
from lerobot_policy_flow_matching import FlowMatchingConfig, FlowMatchingPolicy


def test_predict_action_chunk_shape_offline(
    policy: FlowMatchingPolicy, config: FlowMatchingConfig, train_batch: dict[str, torch.Tensor]
):
    """predict_action_chunk() on an offline batch returns (B, n_action_steps, action_dim)."""
    policy.reset()
    chunk = policy.predict_action_chunk(train_batch)
    assert chunk.shape == (BATCH_SIZE, config.n_action_steps, ACTION_DIM)
    assert torch.isfinite(chunk).all()


def test_select_action_shape(policy: FlowMatchingPolicy, online_batch: dict[str, torch.Tensor]):
    """select_action() on a single-timestep online batch returns one action per item in the batch."""
    policy.reset()
    action = policy.select_action(dict(online_batch))
    assert action.shape == (BATCH_SIZE, ACTION_DIM)
    assert torch.isfinite(action).all()


def test_select_action_reuses_chunk_within_n_action_steps(
    policy: FlowMatchingPolicy, config: FlowMatchingConfig, online_batch: dict[str, torch.Tensor]
):
    """A new action chunk is only generated once every n_action_steps calls (receding-horizon caching)."""
    policy.reset()

    with patch.object(
        policy.flow_matching, "generate_actions", wraps=policy.flow_matching.generate_actions
    ) as mock_generate_actions:
        for _ in range(config.n_action_steps):
            policy.select_action(dict(online_batch))
        assert mock_generate_actions.call_count == 1, "one chunk should cover exactly n_action_steps calls"

        policy.select_action(dict(online_batch))
        assert mock_generate_actions.call_count == 2, (
            "the (n_action_steps + 1)-th call should trigger a new chunk"
        )


def test_select_action_runs_many_steps_without_error(
    policy: FlowMatchingPolicy, config: FlowMatchingConfig, online_batch: dict[str, torch.Tensor]
):
    """select_action() keeps working across multiple chunk-refill cycles."""
    policy.reset()
    for _ in range(2 * config.n_action_steps + 1):
        action = policy.select_action(dict(online_batch))
        assert action.shape == (BATCH_SIZE, ACTION_DIM)
