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
"""Flow Matching policy: a drop-in, generative-process-only variant of lerobot's DiffusionPolicy.

Reuses `DiffusionRgbEncoder` and `DiffusionConditionalUnet1d` directly from
`lerobot.policies.diffusion.modeling_diffusion` (safe to import without pulling in `diffusers` -- that
dependency only lives behind `DiffusionPolicy.__init__`/`_make_noise_scheduler`, never inside the
encoder/UNet classes themselves) and swaps the DDPM/DDIM forward-diffusion + reverse `.step()` loop for the
linear-interpolant forward process + forward-Euler ODE reverse loop from
`lerobot.policies.common.flow_matching`. See `docs/plan.md` for the full design rationale.
"""

from collections import deque

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from lerobot.policies import PreTrainedPolicy
from lerobot.policies.common.flow_matching import euler_integrate, sample_noise, sample_time_beta
from lerobot.policies.diffusion.modeling_diffusion import DiffusionConditionalUnet1d, DiffusionRgbEncoder
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters, populate_queues
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from torch import Tensor, nn

from .configuration_flow_matching import FlowMatchingConfig


class FlowMatchingPolicy(PreTrainedPolicy):
    """Flow Matching policy for visuomotor control, as described in `docs/plan.md`.

    Architecturally identical to lerobot's `DiffusionPolicy` (ResNet + SpatialSoftmax vision encoder, 1D
    conv UNet, receding-horizon action-chunk execution); only the generative process differs.
    """

    config_class = FlowMatchingConfig
    name = "flow_matching"

    def __init__(
        self,
        config: FlowMatchingConfig,
        **kwargs,
    ):
        """Build the policy from `config`. `config.validate_features()` is called here explicitly."""
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.flow_matching = FlowMatchingModel(config)

        # Populated by reset() below; contains the n latest observations and actions during rollout.
        self.reset()

    def get_optim_params(self) -> dict:
        """Return the parameters to pass to the optimizer.

        `PreTrainedPolicy.get_optim_params` is annotated `-> dict`, but a bare parameter iterator is a
        valid (and, for a single optimizer group, the normal) argument to `torch.optim.Adam` -- this is
        the same pattern `DiffusionPolicy.get_optim_params` uses.
        """
        return self.flow_matching.parameters()  # type: ignore[return-value]

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`."""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Predict a chunk of actions given environment observations.

        Supports two modes:
        - Online (queues populated via select_action): stacks observations from internal queues.
        - Offline (empty queues, e.g. dataloader batch): uses the batch directly.
        """
        queues_populated = any(len(q) > 0 for q in self._queues.values())
        if queues_populated:
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        else:
            batch = dict(batch)
            if self.config.image_features:
                for key in self.config.image_features:
                    if batch[key].ndim == 4:
                        batch[key] = batch[key].unsqueeze(1)
                batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        actions = self.flow_matching.generate_actions(batch, noise=noise)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying flow-matching model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The flow-matching model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... | n-o+h |
            |observation is used | YES   | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps <= horizon - n_obs_steps + 1`.
        """
        # NOTE: for offline evaluation, we have action in the batch, so we need to pop it out
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        # NOTE: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the flow-matching loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            for key in self.config.image_features:
                if self.config.n_obs_steps == 1 and batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        loss = self.flow_matching.compute_loss(batch)
        # no output_dict so returning None
        return loss, None


class FlowMatchingModel(nn.Module):
    """The vision encoder + UNet + flow-matching sampling logic underlying `FlowMatchingPolicy`."""

    def __init__(self, config: FlowMatchingConfig):
        """Build the vision encoder(s) and the conditional UNet (no noise scheduler; see module docstring)."""
        super().__init__()
        self.config = config

        # Build observation encoders (depending on which observations are provided).
        global_cond_dim = self.config.robot_state_feature.shape[0]
        self.rgb_encoder: nn.ModuleList | DiffusionRgbEncoder
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                single_encoder = DiffusionRgbEncoder(config)
                self.rgb_encoder = single_encoder
                global_cond_dim += single_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)

        if config.compile_model:
            self.unet = torch.compile(self.unet, mode=config.compile_mode)

    # ========= inference  ============
    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Sample an action trajectory by integrating the learned velocity field from t=1 to t=0."""
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Sample prior (x_1, the flow-matching noise endpoint).
        sample = (
            noise
            if noise is not None
            else torch.randn(
                size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )

        def denoise_fn(x_t: Tensor, t: Tensor) -> Tensor:
            return self.unet(x_t, t * self.config.time_embed_scale, global_cond=global_cond)

        return euler_integrate(denoise_fn, sample, num_steps=self.config.num_inference_steps)

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and concatenate them all together along with the state vector."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]
        # Extract image features.
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                # Combine batch and sequence dims while rearranging to make the camera index dimension first.
                images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                )
                # Separate batch and sequence dims back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                # Combine batch, sequence, and "which camera" dims before passing to shared encoder.
                img_features = self.rgb_encoder(
                    einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                )
                # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        # Concatenate features then flatten to (B, global_cond_dim).
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Generate a full `horizon`-length action chunk and slice out the `n_action_steps` to execute.

        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, n_obs_steps, environment_dim)
        }
        """
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # run sampling
        actions = self.conditional_sample(batch_size, global_cond=global_cond, noise=noise)

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute the flow-matching MSE loss between the predicted and target velocity fields.

        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, n_obs_steps, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # Flow-matching forward process: linear interpolant between noise (x_1) and the clean action
        # trajectory (x_0), with a constant target velocity (openpi/pi0 convention).
        actions = batch[ACTION]
        noise = sample_noise(actions.shape, actions.device)
        bsize = actions.shape[0]
        time = sample_time_beta(
            bsize,
            actions.device,
            alpha=self.config.time_sampling_alpha,
            beta=self.config.time_sampling_beta,
            scale=self.config.time_sampling_scale,
            offset=self.config.time_sampling_offset,
        )
        t = time[:, None, None]
        noisy_trajectory = t * noise + (1 - t) * actions
        target_velocity = noise - actions

        # Run the denoising network to predict the velocity field.
        pred = self.unet(noisy_trajectory, time * self.config.time_embed_scale, global_cond=global_cond)

        loss = F.mse_loss(pred, target_velocity, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            mask = in_episode_bound.unsqueeze(-1)
            num_valid = mask.sum() * loss.shape[-1]
            return (loss * mask).sum() / num_valid.clamp_min(1)

        return loss.mean()
