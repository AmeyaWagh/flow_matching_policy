#!/usr/bin/env python
"""Standalone smoke test for FlowMatchingPolicy -- no CLI, no dataset download, no gym env.

Builds a policy from synthetic PushT-shaped features, runs a training-style forward pass
(exercises `compute_loss` -> UNet -> loss end-to-end) and both action-generation paths
(`predict_action_chunk` for offline/dataloader-style batches, `select_action` for the
online receding-horizon queue logic exercised during real rollouts). Runs in seconds on
CPU -- iterate here before touching `lerobot-train`/`lerobot-eval`.

Usage:
    python scripts/debug_forward_pass.py
"""

import torch
from lerobot.configs.types import FeatureType, PolicyFeature

from lerobot_policy_flow_matching import FlowMatchingConfig, FlowMatchingPolicy


def main() -> None:
    """Build a policy from synthetic features and exercise forward/predict/select_action."""
    config = FlowMatchingConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
            "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        },
        device="cpu",
    )
    policy = FlowMatchingPolicy(config)
    policy.eval()

    batch_size = 4

    # --- forward() / compute_loss(): training-style batch, all timesteps at once ---
    train_batch = {
        "observation.state": torch.randn(batch_size, config.n_obs_steps, 2),
        "observation.image": torch.rand(batch_size, config.n_obs_steps, 3, 96, 96),
        "action": torch.randn(batch_size, config.horizon, 2),
        "action_is_pad": torch.zeros(batch_size, config.horizon, dtype=torch.bool),
    }
    loss, _ = policy.forward(train_batch)
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    print(f"forward() loss: {loss.item():.4f}")

    # --- predict_action_chunk(): offline/dataloader-style batch (queues empty) ---
    policy.reset()
    chunk = policy.predict_action_chunk(train_batch)
    expected_shape = (batch_size, config.n_action_steps, 2)
    assert chunk.shape == expected_shape, f"expected {expected_shape}, got {chunk.shape}"
    print(f"predict_action_chunk() shape: {tuple(chunk.shape)}")

    # --- select_action(): online receding-horizon queue logic, single-timestep batches ---
    policy.reset()
    online_batch = {
        "observation.state": torch.randn(batch_size, 2),
        "observation.image": torch.rand(batch_size, 3, 96, 96),
    }
    actions = [policy.select_action(dict(online_batch)) for _ in range(config.n_action_steps + 1)]
    for action in actions:
        assert action.shape == (batch_size, 2), f"expected ({batch_size}, 2), got {tuple(action.shape)}"
    print(f"select_action() ran {len(actions)} steps, each shape {tuple(actions[0].shape)}")

    print("OK: all checks passed.")


if __name__ == "__main__":
    main()
