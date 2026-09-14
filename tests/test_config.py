"""Tests for FlowMatchingConfig: field defaults, validation, and delta-index properties."""

import pytest
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.optim import AdamConfig, CosineDecayWithWarmupSchedulerConfig

from conftest import make_config
from lerobot_policy_flow_matching import FlowMatchingConfig


def test_field_defaults():
    """Defaults mirror DiffusionConfig's architecture fields, tuned for apples-to-apples comparison."""
    config = make_config()
    assert config.n_obs_steps == 2
    assert config.horizon == 64
    # 16, not lerobot's current default of 32: an eval-time sweep on identical weights scored
    # 6/52/56/36% for n_action_steps 4/8/16/32 on PushT (see docs/comparison_pusht.md, E1).
    assert config.n_action_steps == 16
    # 7, NOT `horizon - n_action_steps - n_obs_steps + 1` (= 47). 47 collapses PushT success to 2-10%;
    # see the extended rationale in configuration_flow_matching.py and comparison_pusht.md (E4/E4b).
    assert config.drop_n_last_frames == 7
    # Matches the lerobot/diffusion_pusht recipe: random-crop augmentation + a from-scratch GroupNorm
    # backbone (GroupNorm and pretrained weights are mutually exclusive in DiffusionRgbEncoder).
    assert config.crop_shape == (84, 84)
    assert config.crop_is_random is True
    assert config.use_group_norm is True
    assert config.pretrained_backbone_weights is None
    assert config.down_dims == (512, 1024, 2048)
    assert config.diffusion_step_embed_dim == 128
    assert config.num_inference_steps == 10
    assert config.time_sampling_alpha == 1.5
    assert config.time_sampling_beta == 1.0
    assert config.time_embed_scale == 1000.0


def test_validate_features_requires_image_or_env_state():
    """validate_features() rejects a config with neither an image nor an environment-state feature."""
    config = make_config(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        }
    )
    with pytest.raises(ValueError, match="image or the environment state"):
        config.validate_features()


def test_validate_features_requires_matching_image_shapes():
    """validate_features() rejects mismatched shapes across multiple camera features."""
    config = make_config(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
            "observation.image.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
            # Both shapes must stay >= the default (84, 84) crop, or the crop-fit check fires first
            # and this test stops exercising the shape-mismatch check it is named after.
            "observation.image.side": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 88, 88)),
        }
    )
    with pytest.raises(ValueError, match="does not match"):
        config.validate_features()


def test_validate_features_passes_for_default_config():
    """The synthetic PushT-shaped config used across this test suite passes validation."""
    make_config().validate_features()


def test_horizon_must_be_multiple_of_downsampling_factor():
    """__post_init__ rejects a horizon that isn't a multiple of the UNet's downsampling factor."""
    with pytest.raises(ValueError, match="horizon"):
        make_config(horizon=63)  # 2**len((512,1024,2048)) == 8; 63 % 8 != 0


def test_time_sampling_offset_must_be_in_unit_interval():
    """__post_init__ rejects a time_sampling_offset outside (0, 1)."""
    with pytest.raises(ValueError, match="time_sampling_offset"):
        make_config(time_sampling_offset=0.0)


def test_time_sampling_scale_plus_offset_must_not_exceed_one():
    """__post_init__ rejects a scale+offset combination that could push sampled time above 1."""
    with pytest.raises(ValueError, match="time_sampling_scale"):
        make_config(time_sampling_scale=0.999, time_sampling_offset=0.01)


def test_num_inference_steps_must_be_positive():
    """__post_init__ rejects a non-positive num_inference_steps."""
    with pytest.raises(ValueError, match="num_inference_steps"):
        make_config(num_inference_steps=0)


def test_delta_indices():
    """observation/action delta indices encode the receding-horizon windowing, same as DiffusionConfig."""
    config = make_config(n_obs_steps=2, horizon=64)
    assert config.observation_delta_indices == [-1, 0]
    assert config.action_delta_indices == list(range(-1, 63))
    assert config.reward_delta_indices is None


def test_get_optimizer_preset():
    """get_optimizer_preset() returns an AdamConfig matching the configured hyperparameters."""
    config = make_config(optimizer_lr=5e-5)
    preset = config.get_optimizer_preset()
    assert isinstance(preset, AdamConfig)
    assert preset.lr == 5e-5


def test_get_scheduler_preset():
    """get_scheduler_preset() returns a diffusers-free CosineDecayWithWarmupSchedulerConfig."""
    config = make_config(scheduler_warmup_steps=100, scheduler_decay_steps=1000, scheduler_decay_lr=1e-6)
    preset = config.get_scheduler_preset()
    assert isinstance(preset, CosineDecayWithWarmupSchedulerConfig)
    assert preset.num_warmup_steps == 100
    assert preset.num_decay_steps == 1000
    assert preset.decay_lr == 1e-6


def test_registered_under_flow_matching():
    """The config class is registered under the "flow_matching" policy type string."""
    from lerobot.configs import PreTrainedConfig

    assert PreTrainedConfig.get_choice_class("flow_matching") is FlowMatchingConfig
