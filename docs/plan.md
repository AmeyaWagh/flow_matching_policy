# Flow Matching Robot Policy for LeRobot (starting with PushT)

## Context

The goal is a Flow Matching (FM) robot imitation-learning policy pluggable into HuggingFace `lerobot`'s
standard training/eval CLI (`lerobot-train` / `lerobot-eval`), starting with the PushT task and using
`lerobot/diffusion_pusht` (`DiffusionPolicy`) as both the architectural reference and the benchmark to
compare against. The policy must **not** be PushT-specific — like `DiffusionPolicy`, it should work with
any lerobot dataset/env by deriving its architecture from `input_features`/`output_features`.

Key finding driving the whole approach: **lerobot officially supports out-of-tree policy plugins with zero
changes to lerobot itself** — a pip package named `lerobot_policy_<name>` is auto-discovered by
`register_third_party_plugins()`, called unconditionally at the top of both CLIs (documented in lerobot's
`docs/source/bring_your_own_policies.mdx`, which even cites a community flow-matching-objective plugin,
`lerobot_policy_ditflow`, built the same way).

The engineering delta vs. `DiffusionPolicy` is intentionally narrow: same vision encoder, same 1D-conv
UNet, same receding-horizon action-chunk execution — only the noise process changes (DDPM/DDIM
forward-diffusion + `.step()` reverse loop → linear-interpolant forward process + Euler-ODE reverse loop),
using primitives lerobot already ships pure-PyTorch, no `diffusers` dependency
(`lerobot.policies.common.flow_matching`, shared today by pi0/pi05/smolvla/eo1's transformer-based
policies — this project pairs it with the diffusion-policy-style 1D-conv UNet + ResNet/SpatialSoftmax
vision encoder for the first time). Keeping the delta narrow is what makes the eventual FM-vs-diffusion
comparison meaningful rather than a comparison of two unrelated architectures.

See `../CLAUDE.md` for environment/tooling commands and known workarounds; see `checklist.md` for granular,
checkable progress tracking against this plan.

## Repository layout

```
flow_matching_policy/
├── README.md
├── CLAUDE.md
├── pyproject.toml                              # uv-managed; the installable dist IS this package
├── uv.lock
├── .pre-commit-config.yaml
├── LICENSE                                     # Apache-2.0 (building blocks reused from lerobot)
├── src/
│   └── lerobot_policy_flow_matching/           # name is load-bearing: `lerobot_policy_` prefix triggers auto-discovery
│       ├── __init__.py                         # guarded `import lerobot`; imports config+modeling+processor -> registers policy
│       ├── configuration_flow_matching.py      # FlowMatchingConfig(PreTrainedConfig), @register_subclass("flow_matching")
│       ├── modeling_flow_matching.py           # FlowMatchingPolicy(PreTrainedPolicy) + FlowMatchingModel
│       └── processor_flow_matching.py          # make_flow_matching_pre_post_processors
├── tests/
│   ├── test_config.py                          # field defaults, validate_features(), horizon % downsample check
│   ├── test_modeling_forward.py                # synthetic-batch forward() -> finite scalar loss
│   ├── test_modeling_inference.py              # predict_action_chunk() shape; select_action() queue behavior over N calls
│   └── test_plugin_registration.py             # import triggers registration; "flow_matching" in PreTrainedConfig.get_known_choices()
├── scripts/
│   ├── prepare_diffusion_pusht_checkpoint.py   # done — local-loadable copy of the reference checkpoint
│   ├── eval_pretrained.sh                      # done — lerobot-eval wrapper + rollout videos + optional wandb
│   ├── log_eval_to_wandb.py                    # done — pushes eval_info.json + videos to wandb
│   ├── debug_forward_pass.py                   # not yet written — see "Local training/debugging workflow"
│   └── eval_compare.py                         # not yet written — see "Comparison vs. diffusion_pusht"
└── docs/
    ├── plan.md                                 # this file
    ├── checklist.md                            # granular progress checklist
    └── comparison_pusht.md                     # written at M4
```

Single-package `uv` project (not a workspace) — `scripts/`/`tests/`/`docs/` are plain root-level dirs; the
only pip-installable unit is `lerobot_policy_flow_matching`, matching lerobot's mandated plugin layout
exactly.

## Config class (`configuration_flow_matching.py`)

Mirror `lerobot.policies.diffusion.configuration_diffusion.DiffusionConfig` field-for-field:

- **Keep verbatim** (names matter — some are read by attribute from reused lerobot classes, see below):
  `n_obs_steps=2`, `horizon=64`, `n_action_steps=32`, `normalization_mapping` (VISUAL:MEAN_STD, STATE:MIN_MAX,
  ACTION:MIN_MAX), `drop_n_last_frames=7`; vision-backbone fields (`vision_backbone="resnet18"`, `resize_shape`,
  `crop_ratio`, `crop_shape`, `crop_is_random`, `pretrained_backbone_weights`, `use_group_norm`,
  `spatial_softmax_num_keypoints=32`, `use_separate_rgb_encoder_per_camera=True`); UNet fields
  (`down_dims=(512,1024,2048)`, `kernel_size=5`, `n_groups=8`, `diffusion_step_embed_dim=128` — kept under
  this name because `DiffusionConditionalUnet1d.__init__` reads `config.diffusion_step_embed_dim` by
  attribute, `use_film_scale_modulation=True`, `gradient_checkpointing`, `compile_model`/`compile_mode`,
  `do_mask_loss_for_padding=False`).
- **Drop** (diffusion-noise-process-specific, no FM analogue): `noise_scheduler_type`, `num_train_timesteps`,
  `beta_schedule`, `beta_start`, `beta_end`, `prediction_type`, `clip_sample`, `clip_sample_range`.
- **Add** (FM-specific): `num_inference_steps: int = 10` (pi0-style default vs. diffusion's ~100 — tune at
  M2), `time_sampling_alpha: float = 1.5`, `time_sampling_beta: float = 1.0`, `time_sampling_scale: float =
  0.999`, `time_sampling_offset: float = 0.001` (Beta-distributed training-time sampler, matches pi0/pi05
  defaults), `time_embed_scale: float = 1000.0` (see below), `scheduler_decay_steps: int = 200_000`,
  `scheduler_decay_lr: float = 1e-5`.
- **Optimizer/scheduler**: keep `optimizer_lr=1e-4`, `optimizer_betas=(0.95,0.999)`, `optimizer_eps=1e-8`,
  `optimizer_weight_decay=1e-6`, `scheduler_warmup_steps=500`. `get_optimizer_preset()` returns `AdamConfig`
  identically to `DiffusionConfig`. `get_scheduler_preset()` must return
  `lerobot.optim.CosineDecayWithWarmupSchedulerConfig` (needs `num_warmup_steps`, `num_decay_steps`,
  `peak_lr`, `decay_lr` — all required, no defaults), **not** `DiffuserSchedulerConfig` — the latter calls
  `require_package("diffusers", ...)` and imports `diffusers.optimization.get_scheduler` internally,
  reintroducing the exact dependency this policy otherwise avoids entirely.
- `__post_init__`/`validate_features()`/delta-indices: copy `DiffusionConfig`'s logic verbatim (including the
  `horizon % 2**len(down_dims) == 0` check — still applies, same UNet) minus the removed noise-scheduler
  validation. `validate_features()` is **not called automatically by the base class** — `FlowMatchingPolicy.__init__`
  must call it manually, same as `DiffusionPolicy.__init__` does.

## Model (`modeling_flow_matching.py`)

**Reuse, don't vendor:** import `DiffusionRgbEncoder` and `DiffusionConditionalUnet1d` directly from
`lerobot.policies.diffusion.modeling_diffusion`. This is safe — `diffusers` is only referenced in that
module's scheduler-import block (`TYPE_CHECKING`/`_diffusers_available`-gated, falls back to `None`) and
inside `DiffusionPolicy.__init__`/`_make_noise_scheduler`, never inside the encoder/UNet classes themselves.
Importing just those classes pulls in no `diffusers` dependency, keeps the diff against the reference
architecture minimal (the whole point of the comparison), and means future lerobot fixes to those classes
are inherited for free. Trade-off: couples us to lerobot internals not marked as stable public API — pin a
specific lerobot version in `pyproject.toml` (already done: `>=0.6.1,<0.7`) and rely on
`tests/test_modeling_forward.py` to catch any breakage loudly. Fall back to vendoring only if this actually
breaks upstream.

```python
from lerobot.policies.diffusion.modeling_diffusion import DiffusionRgbEncoder, DiffusionConditionalUnet1d
from lerobot.policies.common.flow_matching import sample_noise, sample_time_beta, euler_integrate
```

`FlowMatchingModel.__init__(config)`: identical `global_cond_dim` computation to `DiffusionModel.__init__`
(per-camera `DiffusionRgbEncoder`(s) + state [+ env_state]), builds
`self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=...)`. **No noise scheduler** — that's the
entire structural delta in `__init__`.

**Training (`compute_loss`)** — linear interpolant + constant-velocity target (openpi/pi0 convention):

```python
global_cond = self._prepare_global_conditioning(batch)  # copy verbatim from DiffusionModel
actions = batch[ACTION]  # (B, horizon, action_dim), the clean sample x_0
noise = sample_noise(actions.shape, actions.device)  # x_1
time = sample_time_beta(
    bsize,
    device,
    alpha=config.time_sampling_alpha,
    beta=config.time_sampling_beta,
    scale=config.time_sampling_scale,
    offset=config.time_sampling_offset,
)  # (B,) in (0,1)
t = time[:, None, None]
x_t = t * noise + (1 - t) * actions
u_t = noise - actions
pred_v = self.unet(x_t, time * config.time_embed_scale, global_cond=global_cond)
loss = F.mse_loss(pred_v, u_t, reduction="none")
# same do_mask_loss_for_padding masking as DiffusionModel.compute_loss
```

`DiffusionConditionalUnet1d.forward(x, timestep, global_cond)` does not cast `timestep` to `.long()`
anywhere — it flows straight into `Sequential(DiffusionSinusoidalPosEmb, Linear, Mish, Linear)`, and
`DiffusionSinusoidalPosEmb.forward` is a plain float multiply. Feeding it a continuous float `time` tensor
is safe as-is; no changes to the reused UNet are needed.

**Why `time_embed_scale=1000.0`:** `DiffusionSinusoidalPosEmb` was designed for integer diffusion timesteps
in `[0,100)`; feeding it raw `t ∈ (0,1)` directly would keep even its highest-frequency channel under one
full sine period, collapsing most of the embedding's resolving power. Scaling `t * 1000` before embedding
(the same convention SD3/Flux use when reusing a discrete-timestep embedding for continuous time) fixes
this with a one-line change and keeps the UNet 100% parameter-identical to `diffusion_pusht`. Validate at M2
(loss should decrease smoothly); if not, sweep `time_embed_scale ∈ {100, 1000, 10000}` before suspecting
anything else.

**Sampling (`conditional_sample`)** — replaces `DiffusionModel.conditional_sample`'s DDPM/DDIM loop:

```python
def denoise_fn(x_t, t):
    return self.unet(x_t, t * config.time_embed_scale, global_cond=global_cond)


return euler_integrate(denoise_fn, noise, num_steps=config.num_inference_steps)
```

`generate_actions()` is identical to diffusion's: call `conditional_sample`, then slice
`actions[:, n_obs_steps-1 : n_obs_steps-1+n_action_steps]`.

**`FlowMatchingPolicy(PreTrainedPolicy)`** — copy `DiffusionPolicy`'s structure near-verbatim: `config_class`/
`name = "flow_matching"`; `__init__` has **no `require_package` call** (nothing to gate — this is the one
meaningful deletion vs. diffusion's `__init__`), calls `config.validate_features()` manually, builds
`self.flow_matching = FlowMatchingModel(config)`, calls `self.reset()`; `get_optim_params()` returns
`self.flow_matching.parameters()`; `reset()` builds the same observation/action deques keyed by
`OBS_STATE`/`ACTION`/`OBS_IMAGES`/`OBS_ENV_STATE`; `predict_action_chunk()`/`select_action()` copy the
receding-horizon queue logic exactly (online queues-populated mode vs. offline dataloader-batch mode; the
`ACTION in batch: batch.pop(ACTION)` guard; `populate_queues` called after image stacking); `forward()`
stacks per-camera images into `OBS_IMAGES`, delegates to `compute_loss`, returns `(loss, None)`.

## Processor (`processor_flow_matching.py`)

Trivial delegate, identical pattern to `processor_diffusion.py`:

```python
def make_flow_matching_pre_post_processors(config, dataset_stats=None):
    return make_default_pre_post_processors(config, dataset_stats)
```

`__init__.py` follows lerobot's out-of-tree plugin template: guarded `import lerobot` (raise a clear
`ImportError` if missing), then import+re-export `FlowMatchingConfig`, `FlowMatchingPolicy`,
`make_flow_matching_pre_post_processors` in `__all__`. (Unlike in-tree policies, out-of-tree plugins *do*
eagerly re-export the modeling class — there's no "keep `import lerobot` fast" constraint here.)

## Local training/debugging workflow

Before any real training run, add `scripts/debug_forward_pass.py`: build a `FlowMatchingConfig` with
synthetic PushT-shaped `PolicyFeature`s (state `(2,)`, image `(3,384,384)`, action `(2,)` — see
`lerobot.envs.configs.PushtEnv` for the real feature keys/shapes), instantiate `FlowMatchingPolicy`, run
`forward()` on a synthetic batch (exercises `compute_loss` → UNet → loss end-to-end) and
`predict_action_chunk()` (exercises the `euler_integrate` sampling path). No CLI, no dataset download — runs
in seconds on CPU. Iterate here before touching `lerobot-train`.

Then the milestone commands (M1–M2 below), and finally:

```bash
lerobot-train --policy.type=flow_matching --dataset.repo_id=lerobot/pusht --env.type=pusht \
  --steps=200000 --batch_size=64 --wandb.enable=true --output_dir=outputs/train/flow_matching_pusht
```

## Comparison vs. `lerobot/diffusion_pusht`

Both policies go through the identical `eval_policy_all` (reports `pc_success`, `avg_sum_reward`, writes
`eval_info.json`) — no new eval framework needed, just two `lerobot-eval` invocations with the same
`--eval.n_episodes`/`--seed`, one per `--policy.path`. `scripts/eval_compare.py` (not yet written) should
load both `eval_info.json` files and print a small diff table — keep it to ~30 lines, not a new harness.

## Generalization beyond PushT

Nothing in the design above hardcodes PushT: `global_cond_dim` is derived from
`config.robot_state_feature`/`config.image_features`/`config.env_state_feature`, the UNet's action-channel
width comes from `config.action_feature.shape[0]`, and the receding-horizon queue logic is pure sequence
windowing. Only the *defaults* (`down_dims`, `horizon`, `drop_n_last_frames`, native `384×384` resolution)
are PushT-tuned — same as `DiffusionConfig` itself. A second task needs only
`--dataset.repo_id=...`/`--env.type=...` and possibly architecture-capacity CLI overrides. One inherited
(non-FM-specific) limitation: all camera images must share the same shape (enforced in `validate_features()`,
inherited from the reused vision pipeline).

## Milestones and validation gates

1. **M1 — Scaffold + registration.** Validate:
   `python -c "import lerobot_policy_flow_matching; from lerobot.configs import PreTrainedConfig; assert 'flow_matching' in PreTrainedConfig.get_known_choices()"`,
   `scripts/debug_forward_pass.py` runs clean, then
   `lerobot-train --policy.type=flow_matching --dataset.repo_id=lerobot/pusht --env.type=pusht --steps=1 --batch_size=8 --output_dir=outputs/train/m1_smoke --wandb.enable=false`
   produces a checkpoint with no shape/import errors.
2. **M2 — Short training run, decreasing loss.** `--steps=5000 --batch_size=64`; FM MSE loss should trend
   down with no NaNs. If flat/noisy, sweep `time_embed_scale` first.
3. **M3 — Nonzero rollout success on PushT.**
   `lerobot-eval --policy.path=outputs/train/m2_short/checkpoints/last/pretrained_model --env.type=pusht --eval.n_episodes=20 --eval.use_async_envs=false`.
   This is the first point `reset()`/`select_action()`/queues get exercised in a real env rollout rather than
   offline dataloader batches.
4. **M4 — Full run + side-by-side comparison.** Full 200k-step run, `--eval.n_episodes=50` on both policies,
   `scripts/eval_compare.py` table into `docs/comparison_pusht.md`. Not required to beat diffusion — the
   goal is a correct, fairly-compared FM policy; a wildly-worse result (e.g. near-zero success) is a
   correctness-bug signal, not an accepted conclusion.

## Critical files for reference during implementation

- `lerobot.policies.diffusion.modeling_diffusion` — source of reused classes (`DiffusionRgbEncoder`,
  `DiffusionConditionalUnet1d`) + structural template for `FlowMatchingPolicy`/`FlowMatchingModel`.
- `lerobot.policies.diffusion.configuration_diffusion` — field-by-field template for `FlowMatchingConfig`.
- `lerobot.policies.common.flow_matching` — `sample_noise`/`sample_time_beta`/`euler_integrate`, import
  directly, no vendoring needed.
- `lerobot.optim.schedulers` — `CosineDecayWithWarmupSchedulerConfig` (diffusers-free scheduler).
- `lerobot.envs.configs.PushtEnv` — PushT feature keys/shapes.
- lerobot's `docs/source/bring_your_own_policies.mdx` — authoritative out-of-tree plugin template.
