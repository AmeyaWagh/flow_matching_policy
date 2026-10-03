"""Tests for the training-time timestep samplers in time_samplers.py."""

import pytest
import torch

from conftest import make_config
from lerobot_policy_flow_matching.time_samplers import sample_time, sample_time_logit_normal


def test_logit_normal_shape_and_range():
    """sample_time_logit_normal returns (bsize,) values strictly within (0, 1)."""
    t = sample_time_logit_normal(1000, torch.device("cpu"), mean=-1.0, std=1.0)
    assert t.shape == (1000,)
    assert torch.all(t > 0) and torch.all(t < 1)


def test_logit_normal_negative_mean_biases_toward_zero():
    """A strongly negative mean concentrates most samples below 0.5 (toward the target action, t=0)."""
    t = sample_time_logit_normal(2000, torch.device("cpu"), mean=-2.0, std=0.5)
    assert (t < 0.5).float().mean() > 0.9


def test_logit_normal_positive_mean_biases_toward_one():
    """A strongly positive mean concentrates most samples above 0.5 (toward noise, t=1)."""
    t = sample_time_logit_normal(2000, torch.device("cpu"), mean=2.0, std=0.5)
    assert (t > 0.5).float().mean() > 0.9


def test_sample_time_dispatches_to_beta():
    """sample_time("beta", ...) returns values in (offset, offset + scale] as sample_time_beta does."""
    config = make_config(time_sampling_distribution="beta")
    t = sample_time("beta", 500, torch.device("cpu"), config)
    assert t.shape == (500,)
    assert torch.all(t > 0) and torch.all(t <= 1)


def test_sample_time_dispatches_to_logit_normal():
    """sample_time("logit_normal", ...) uses config's mean/std fields."""
    config = make_config(
        time_sampling_distribution="logit_normal",
        time_sampling_logit_normal_mean=-1.0,
        time_sampling_logit_normal_std=1.0,
    )
    t = sample_time("logit_normal", 500, torch.device("cpu"), config)
    assert t.shape == (500,)
    assert torch.all(t > 0) and torch.all(t < 1)


def test_sample_time_rejects_unknown_distribution():
    """sample_time raises for a distribution name outside TIME_SAMPLING_DISTRIBUTIONS."""
    config = make_config()
    with pytest.raises(ValueError, match="Unknown"):
        sample_time("not_a_real_distribution", 10, torch.device("cpu"), config)
