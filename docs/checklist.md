# Implementation checklist

Tracks progress against `plan.md`. Check items off as they land; keep this in sync with reality rather than
with intent — an unchecked item that's actually done is as misleading as the reverse.

## Phase 0 — Environment & tooling

- [x] `uv` venv + `pyproject.toml` (torch/cu128, `lerobot[pusht,training,diffusion]`, `robometric-frame`)
- [x] `.pre-commit-config.yaml` (ruff-check, ruff-format, mypy, interrogate) — all passing, hook installed
- [x] `scripts/prepare_diffusion_pusht_checkpoint.py` — local-loadable copy of the reference checkpoint
- [x] `scripts/eval_pretrained.sh` — `lerobot-eval` wrapper, rollout videos, optional wandb push
- [x] `scripts/log_eval_to_wandb.py` — pushes `eval_info.json` + videos to wandb
- [x] Baseline sanity check: `lerobot/diffusion_pusht` evaluated locally (100% success, 4/4 episodes, smoke run)
- [x] `scripts/train_flow_matching.sh` — `lerobot-train` wrapper (async-env/push-to-hub workarounds baked in,
      positional args for steps/batch/output/dataset/env, `--` passthrough for extra flags)

## Phase 1 — Policy implementation (`src/lerobot_policy_flow_matching/`)

- [x] `configuration_flow_matching.py` — `FlowMatchingConfig(PreTrainedConfig)`, `@register_subclass("flow_matching")`
- [x] `modeling_flow_matching.py` — `FlowMatchingModel(nn.Module)` (reuses `DiffusionRgbEncoder`/`DiffusionConditionalUnet1d`)
- [x] `modeling_flow_matching.py` — `FlowMatchingPolicy(PreTrainedPolicy)` (queues, `select_action`, `predict_action_chunk`, `forward`)
- [x] `processor_flow_matching.py` — `make_flow_matching_pre_post_processors`
- [x] `__init__.py` — guarded `import lerobot`, re-export config/policy/processor factory, `__all__`
- [x] `scripts/debug_forward_pass.py` — standalone synthetic-batch forward/sample smoke test (no CLI, no dataset)

## Phase 2 — Tests (`tests/`)

- [x] `test_config.py` — field defaults, `validate_features()`, `horizon % 2**len(down_dims) == 0` check
- [x] `test_modeling_forward.py` — synthetic-batch `forward()` returns a finite scalar loss
- [x] `test_modeling_inference.py` — `predict_action_chunk()` output shape; `select_action()` queue behavior over N calls
- [x] `test_plugin_registration.py` — importing the package registers `"flow_matching"` in `PreTrainedConfig.get_known_choices()`
- [x] All 21 tests pass; `pre-commit run --all-files` (ruff-check, ruff-format, mypy, interrogate) passes clean

## Phase 3 — Milestones (training/eval on PushT)

- [x] **M1 — Scaffold + registration smoke test**
  - [x] `python -c "import lerobot_policy_flow_matching; ..."` confirms registration
  - [x] `scripts/debug_forward_pass.py` runs clean
  - [x] `lerobot-train --policy.type=flow_matching --dataset.repo_id=lerobot/pusht --env.type=pusht --steps=1
        --batch_size=8 --eval.use_async_envs=false --policy.push_to_hub=false --output_dir=outputs/train/m1_smoke`
        produces a full checkpoint (config, weights, pre/post-processors), no shape/import errors
        (263M params, ~6.4s/step on the RTX 4090)
- [x] **M2 — Short training run, decreasing loss**
  - [x] `scripts/train_flow_matching.sh 5000 64 outputs/train/m2_short` run completes (~6.5 min on the RTX 4090,
        13.4 steps/s)
  - [x] FM MSE loss trends down smoothly and monotonically: 0.487 (step 50) → 0.077 (step 500) → 0.056
        (step 1K) → 0.035 (step 2K) → 0.017 (step 4K) → 0.014 (step 5K), ~35x reduction. Gradient norm
        shrinks in step (6.37 → 0.30) and stays bounded throughout. Zero NaN/Inf/error occurrences in the
        log. `time_embed_scale=1000.0` worked on the first try -- no sweep needed.
- [ ] **M3 — Nonzero rollout success on PushT**
  - [ ] `lerobot-eval` against the M2 checkpoint (`--eval.n_episodes=20 --eval.use_async_envs=false`) reports `pc_success > 0`
- [ ] **M4 — Full run + side-by-side comparison**
  - [ ] Full `--steps=200000` training run to convergence
  - [ ] `lerobot-eval --eval.n_episodes=50` on both the FM checkpoint and `models/diffusion_pusht_local`
  - [ ] `scripts/eval_compare.py` written (diffs the two `eval_info.json` files)
  - [ ] Results + repro commands written to `docs/comparison_pusht.md`

## Phase 4 — Generalization (stretch, after PushT is solid)

- [ ] Confirm a second lerobot env/dataset (e.g. ALOHA) works with only `--dataset.repo_id`/`--env.type`
      overrides — verify feature keys/shapes first (`grep -n "class AlohaEnv" -A 30` in lerobot's
      `envs/configs.py`) since this wasn't checked during the PushT-focused planning pass
- [ ] Note any PushT-specific leakage found and fix it
