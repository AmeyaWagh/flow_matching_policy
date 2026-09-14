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
"""Configuration for FlowMatchingPolicy.

Mirrors `lerobot.policies.diffusion.configuration_diffusion.DiffusionConfig` field-for-field so the two
policies are directly comparable on the same task (see `docs/plan.md` for the full rationale). The only
architectural difference is the generative process: continuous-time flow matching (linear-interpolant
forward process + forward-Euler ODE sampling) instead of discrete-time denoising diffusion (DDPM/DDIM).
"""

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamConfig, CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("flow_matching")
@dataclass
class FlowMatchingConfig(PreTrainedConfig):
    """Configuration class for FlowMatchingPolicy.

    Defaults are configured for training with PushT providing proprioceptive and single camera
    observations, matching `lerobot/diffusion_pusht`'s architecture for apples-to-apples comparison.

    Notes on the inputs and outputs (same contract as DiffusionConfig):
        - "observation.state" is required as an input key.
        - Either:
            - At least one key starting with "observation.image" is required as an input,
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.image" they are treated as multiple
          camera views. Right now we only support all images having the same shape.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        horizon: Flow-matching action prediction size, i.e. the length of the action chunk sampled by one
            call to the ODE solver. See `FlowMatchingPolicy.select_action` for more details.
        n_action_steps: The number of action steps to run in the environment for one invocation of the
            policy. See `FlowMatchingPolicy.select_action` for more details.
        input_features: A dictionary defining the PolicyFeature of the input data for the policy.
        output_features: A dictionary defining the PolicyFeature of the output data for the policy.
        normalization_mapping: A dictionary that maps from a str value of FeatureType (e.g., "STATE",
            "VISUAL") to a corresponding NormalizationMode (e.g., NormalizationMode.MIN_MAX).
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        resize_shape: (H, W) shape to resize images to as a preprocessing step for the vision backbone.
            If None, no resizing is done and the original image resolution is used.
        crop_ratio: Ratio in (0, 1] used to derive the crop size from resize_shape. Set to 1.0 to disable
            cropping. Only takes effect when resize_shape is not None.
        crop_shape: (H, W) shape to crop images to. Computed automatically when resize_shape is set and
            crop_ratio < 1.0; can also be set directly for legacy configs that use crop-only.
        crop_is_random: Whether the crop should be random at training time (always a center crop in eval).
        pretrained_backbone_weights: Pretrained torchvision weights to initialize the backbone. `None`
            means no pretrained weights.
        use_group_norm: Whether to replace batch normalization with group normalization in the backbone.
        spatial_softmax_num_keypoints: Number of keypoints for SpatialSoftmax.
        use_separate_rgb_encoder_per_camera: Whether to use a separate RGB encoder for each camera view.
        down_dims: Feature dimension for each stage of temporal downsampling in the UNet.
        kernel_size: The convolutional kernel size of the UNet.
        n_groups: Number of groups used in the group norm of the UNet's convolutional blocks.
        diffusion_step_embed_dim: The UNet is conditioned on the (continuous) flow-matching time via a
            small non-linear network; this is that network's output dimension. Named to match the reused
            `DiffusionConditionalUnet1d` constructor, which reads this field by attribute name.
        use_film_scale_modulation: FiLM (https://huggingface.co/papers/1709.07871) is used for the UNet
            conditioning. Bias modulation is used by default; this also enables scale modulation.
        gradient_checkpointing: Whether to checkpoint the UNet residual blocks during training.
        num_inference_steps: Number of forward-Euler ODE steps to use at inference time.
        time_sampling_alpha: Alpha parameter of the Beta(alpha, beta) distribution used to sample training
            timesteps (openpi/pi0 convention).
        time_sampling_beta: Beta parameter of the Beta(alpha, beta) distribution used to sample training
            timesteps.
        time_sampling_scale: Scale applied to the Beta(alpha, beta) sample before adding
            `time_sampling_offset`, so that `time ~ Beta(alpha, beta) * scale + offset`.
        time_sampling_offset: Offset added after scaling; keeps `time` away from the exact 0/1 boundary.
        time_embed_scale: Multiplier applied to the continuous time `t in (0, 1)` before it is fed into the
            (reused, unmodified) `DiffusionSinusoidalPosEmb` embedding. That embedding was designed for
            integer diffusion timesteps in `[0, num_train_timesteps)`; without this scaling, its
            higher-frequency channels never complete a full period over `t in (0, 1)` and most of the
            embedding's capacity goes unused. Scaling by ~1000 (matching the convention used by other
            codebases that reuse a discrete-timestep sinusoidal embedding for continuous time, e.g. SD3/
            Flux) restores that resolving power without changing the UNet architecture at all.
        compile_model: Whether to `torch.compile` the UNet.
        compile_mode: `torch.compile` mode to use when `compile_model` is True.
        do_mask_loss_for_padding: Whether to mask the loss when there are copy-padded actions. See
            `LeRobotDataset` and `load_previous_and_future_frames` for more information.
        log_trajectory_metrics: Whether to compute trajectory-quality diagnostics (path length,
            path smoothness; via `robometric_frame`) and surface them in `forward`'s `output_dict`.
            `lerobot-train` logs `output_dict` automatically (console + wandb), so this needs no
            extra wiring. Ground-truth metrics (over `batch[ACTION]`) are cheap and computed every
            call; predicted-trajectory metrics need extra Euler-ODE sampling passes, so they're
            gated by `trajectory_metrics_log_freq`.
        trajectory_metrics_log_freq: How often (in training steps) to additionally sample the model's
            own predicted trajectory and compute its path length/smoothness, for comparison against
            the ground-truth values. Only takes effect during training (not the periodic offline
            eval-loss pass) and when `log_trajectory_metrics` is True.
        optimizer_lr: Learning rate for the Adam optimizer preset.
        optimizer_betas: Adam beta coefficients.
        optimizer_eps: Adam epsilon.
        optimizer_weight_decay: Adam weight decay.
        scheduler_warmup_steps: Number of linear warmup steps for the cosine-decay-with-warmup scheduler.
        scheduler_decay_steps: Number of steps over which the learning rate decays to `scheduler_decay_lr`.
            Should typically match (or be close to) the total number of training steps.
        scheduler_decay_lr: Learning rate reached at the end of the cosine decay.
    """

    # Inputs / output structure.
    n_obs_steps: int = 2
    horizon: int = 64
    n_action_steps: int = 16

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Number of end-of-episode frames excluded as window *start* indices by lerobot's
    # `EpisodeAwareSampler`, to avoid training on excessively padded action targets.
    #
    # Keep this at 7. It is NOT `horizon - n_action_steps - n_obs_steps + 1` (= 47 at the current
    # defaults), despite what older comments here and in lerobot claimed: that formula dates from the
    # original diffusion-policy config (horizon=16, n_action_steps=8, n_obs_steps=2 -> 7) and does not
    # generalize to horizon=64. lerobot's own `DiffusionConfig` likewise still ships 7 alongside
    # horizon=64 / n_action_steps=32.
    #
    # Measured, not assumed: a 50k-step run with 47 collapsed PushT rollout success to 2-10% (50 eps)
    # while an otherwise-identical run with 7 reached 62% -- see `docs/comparison_pusht.md` (E4/E4b).
    # With 47, ~40% of each ~125-frame PushT episode is removed from the sampler's start indices, so the
    # terminal fine-alignment phase is only ever seen in chunk positions 17-64 and never in positions
    # 0-15, which is the only slice receding-horizon execution ever runs. The symptom was a pile-up of
    # episodes just below the success threshold (20/50 in max_reward [0.90, 0.95)) rather than an
    # outright-broken policy.
    drop_n_last_frames: int = 7

    # Architecture / modeling.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    resize_shape: tuple[int, int] | None = None
    crop_ratio: float = 1.0
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    # `use_group_norm=True` requires training the backbone from scratch (see `DiffusionRgbEncoder`):
    # replacing BatchNorm with GroupNorm in a pretrained backbone would ruin its pretrained weights.
    # `lerobot/diffusion_pusht` itself uses this combination (pretrained_backbone_weights=None).
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = True
    # UNet.
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True
    gradient_checkpointing: bool = False

    # Flow matching.
    num_inference_steps: int = 10
    time_sampling_alpha: float = 1.5
    time_sampling_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    time_embed_scale: float = 1000.0

    # Optimization
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"

    # Loss computation
    do_mask_loss_for_padding: bool = False

    # Trajectory-quality monitoring (via robometric_frame)
    log_trajectory_metrics: bool = True
    trajectory_metrics_log_freq: int = 200

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_warmup_steps: int = 500
    scheduler_decay_steps: int = 200_000
    scheduler_decay_lr: float = 1e-5

    def __post_init__(self):
        """Validate fields and derive `crop_shape` from `resize_shape`/`crop_ratio` when applicable."""
        super().__post_init__()

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        if not 0.0 < self.time_sampling_offset < 1.0:
            raise ValueError(f"`time_sampling_offset` must be in (0, 1). Got {self.time_sampling_offset}.")
        if self.time_sampling_scale + self.time_sampling_offset > 1.0:
            raise ValueError(
                "`time_sampling_scale + time_sampling_offset` must be <= 1 so sampled time stays in "
                f"(0, 1]. Got scale={self.time_sampling_scale}, offset={self.time_sampling_offset}."
            )
        if self.num_inference_steps < 1:
            raise ValueError(f"`num_inference_steps` must be >= 1. Got {self.num_inference_steps}.")
        if self.trajectory_metrics_log_freq < 1:
            raise ValueError(
                f"`trajectory_metrics_log_freq` must be >= 1. Got {self.trajectory_metrics_log_freq}."
            )

        if self.resize_shape is not None and (
            len(self.resize_shape) != 2 or any(d <= 0 for d in self.resize_shape)
        ):
            raise ValueError(f"`resize_shape` must be a pair of positive integers. Got {self.resize_shape}.")
        if not (0 < self.crop_ratio <= 1.0):
            raise ValueError(f"`crop_ratio` must be in (0, 1]. Got {self.crop_ratio}.")

        if self.resize_shape is not None:
            if self.crop_ratio < 1.0:
                self.crop_shape = (
                    int(self.resize_shape[0] * self.crop_ratio),
                    int(self.resize_shape[1] * self.crop_ratio),
                )
            else:
                # Explicitly disable cropping for resize+ratio path when crop_ratio == 1.0.
                self.crop_shape = None
        if self.crop_shape is not None and (self.crop_shape[0] <= 0 or self.crop_shape[1] <= 0):
            raise ValueError(f"`crop_shape` must have positive dimensions. Got {self.crop_shape}.")

        # Check that the horizon size and U-Net downsampling is compatible.
        # U-Net downsamples by 2 with each stage.
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor (which is determined "
                f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )

    def get_optimizer_preset(self) -> AdamConfig:
        """Return the Adam optimizer preset used to train this policy."""
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        """Return the LR scheduler preset used to train this policy.

        Deliberately not `DiffuserSchedulerConfig` (diffusion's choice) -- that class imports
        `diffusers.optimization.get_scheduler` internally, which would reintroduce the `diffusers`
        dependency this policy otherwise avoids entirely.
        """
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
        )

    def validate_features(self) -> None:
        """Validate input/output feature compatibility. Not called automatically by the base class."""
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        if self.resize_shape is None and self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the image shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for `{key}`."
                    )

        # Check that all input images have the same shape.
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all image shapes "
                        "to match."
                    )

    @property
    def observation_delta_indices(self) -> list:
        """Relative timestep offsets the dataset loader provides per observation."""
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        """Relative timestep offsets for the action chunk the dataset loader returns."""
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        """This policy does not consume rewards."""
        return None
