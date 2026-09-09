"""Shared pytest fixtures for lerobot_policy_flow_matching tests."""

from typing import Any

import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature

from lerobot_policy_flow_matching import FlowMatchingConfig, FlowMatchingPolicy

STATE_DIM = 2
ACTION_DIM = 2
IMAGE_SHAPE = (3, 96, 96)
BATCH_SIZE = 4


def make_config(**overrides: Any) -> FlowMatchingConfig:
    """Build a `FlowMatchingConfig` with synthetic PushT-shaped features, small enough to run fast on CPU."""
    kwargs: dict[str, Any] = {
        "input_features": {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=IMAGE_SHAPE),
        },
        "output_features": {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
        },
        "device": "cpu",
    }
    kwargs.update(overrides)
    return FlowMatchingConfig(**kwargs)


@pytest.fixture
def config() -> FlowMatchingConfig:
    """A default `FlowMatchingConfig` with synthetic PushT-shaped features."""
    return make_config()


@pytest.fixture
def policy(config: FlowMatchingConfig) -> FlowMatchingPolicy:
    """A `FlowMatchingPolicy` built from the default `config` fixture, in eval mode."""
    p = FlowMatchingPolicy(config)
    p.eval()
    return p


@pytest.fixture
def train_batch(config: FlowMatchingConfig) -> dict[str, torch.Tensor]:
    """A synthetic training-style batch: all `n_obs_steps`/`horizon` timesteps at once."""
    return {
        "observation.state": torch.randn(BATCH_SIZE, config.n_obs_steps, STATE_DIM),
        "observation.image": torch.rand(BATCH_SIZE, config.n_obs_steps, *IMAGE_SHAPE),
        "action": torch.randn(BATCH_SIZE, config.horizon, ACTION_DIM),
        "action_is_pad": torch.zeros(BATCH_SIZE, config.horizon, dtype=torch.bool),
    }


@pytest.fixture
def online_batch() -> dict[str, torch.Tensor]:
    """A synthetic single-timestep batch, as `select_action` receives during a real env rollout."""
    return {
        "observation.state": torch.randn(BATCH_SIZE, STATE_DIM),
        "observation.image": torch.rand(BATCH_SIZE, *IMAGE_SHAPE),
    }
