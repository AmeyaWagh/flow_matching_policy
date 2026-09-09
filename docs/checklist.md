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
- [x] **M3 — Nonzero rollout success on PushT**
  - [x] Investigated the 5K-step (M2) checkpoint's `pc_success=0` result first: `avg_max_reward=0.626`,
        individual episodes hit 0.985/0.995 max reward, and a frame-by-frame video check (episode 7)
        showed the policy pushing the block into near-perfect alignment mid-episode before drifting
        just outside `gym_pusht`'s 95%-coverage success threshold -- confirmed undertraining, not a bug.
  - [x] Extended training to 20K steps (`scripts/train_flow_matching.sh 30000 ...`, salvaged from a
        checkpoint saved just before an unrelated periodic-eval crash -- see `train_flow_matching.sh`'s
        header comment on `--env_eval_freq=0` vs the broken `--eval.n_episodes=0`)
  - [x] `scripts/eval_pretrained.sh outputs/train/m3_extended/checkpoints/020000/pretrained_model pusht 20`
        reports `pc_success=10.0` (2/20 episodes) and `avg_max_reward=0.74` -- real, unambiguous successes
  - [x] Resumed to the full 30K-step target (`--config_path=.../checkpoints/020000/.../train_config.json
        --resume=true --steps=30000 --env_eval_freq=0`, continuing rather than restarting from scratch).
        loss 0.014→0.005, grad norm 0.30→0.14; `pred_path_length`/`pred_path_smoothness` (1.84/0.012)
        converged to nearly match `gt_path_length`/`gt_path_smoothness` (1.92/0.017) -- the trajectory
        metrics tracking real learning progress, not just the loss. Re-eval at 30K:
        `pc_success=15.0` (3/20), `avg_max_reward=0.84` -- improved further over the 20K checkpoint.
- [x] **M4 — Full run + side-by-side comparison**
  - [x] Full `--steps=200000` training run to convergence (`outputs/train/m4_full2`, 4h19m, 200000/200000
        steps, 13.25 steps/s throughout, no interruption; loss 0.487→0.001, grad norm 6.37→0.094;
        `pred_path_length`/`gt_path_length` converged to 1.984/1.923)
  - [x] `lerobot-eval --eval.n_episodes=50` on both the FM checkpoint and `models/diffusion_pusht_local`:
        FM `pc_success=32.0` (16/50), diffusion `pc_success=62.0` (31/50) -- diffusion ahead at equal
        training steps, explained by its ~10x more inference-time sampling steps (100 DDPM vs. 10 Euler),
        not a correctness issue (both reward distributions are coherent, not degenerate; see
        `comparison_pusht.md`)
  - [x] `scripts/eval_compare.py` written and verified against both eval runs; also accepts wandb run
        references (URL or `entity/project/run_id`) as either/both source, verified local-vs-local,
        wandb-vs-wandb, and mixed
  - [x] Results + repro commands + analysis written to `docs/comparison_pusht.md`

## Phase 4 — Generalization (stretch, after PushT is solid)

- [ ] Confirm a second lerobot env/dataset (e.g. ALOHA) works with only `--dataset.repo_id`/`--env.type`
      overrides — verify feature keys/shapes first (`grep -n "class AlohaEnv" -A 30` in lerobot's
      `envs/configs.py`) since this wasn't checked during the PushT-focused planning pass
- [ ] Note any PushT-specific leakage found and fix it
