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
        training steps; not a correctness issue (both reward distributions are coherent, not degenerate;
        see `comparison_pusht.md`)
  - [x] Tested the obvious "more inference steps closes the gap" hypothesis directly
        (`--policy.num_inference_steps=100` on the same checkpoint, no retraining needed) -- it does not
        hold: `pc_success` dropped to 24.0% (12/50) at 100 steps vs. 32.0% at the default 10. Ruled out
        "diffusion just gets more refinement steps" as a full explanation; see `comparison_pusht.md`'s
        "Reading the gap" section for the leading (unconfirmed) hypothesis
  - [x] `scripts/eval_compare.py` written and verified against both eval runs; also accepts wandb run
        references (URL or `entity/project/run_id`) as either/both source, verified local-vs-local,
        wandb-vs-wandb, and mixed
  - [x] Results + repro commands + analysis written to `docs/comparison_pusht.md`

## Phase 3b — M4 follow-up: the 32%-vs-62% gap was configuration, not flow matching

The M4 conclusion above ("correct but less sample-efficient") did not survive follow-up. Full writeup and
all wandb links in `comparison_pusht.md`; summary of what changed and why:

- [x] **E1 — `n_action_steps` was mismatched against the baseline.** The M4 comparison claimed the only
      difference between the two policies was the generative process; pulling both configs showed the FM
      run used lerobot 0.6's *current* `DiffusionConfig` defaults (`horizon=64`, `n_action_steps=32`,
      no crop, `use_group_norm=False`) while the `diffusion_pusht` checkpoint predates them (16/8,
      84x84 crop, GroupNorm). Eval-time sweep on the *unmodified* M4 weights, 50 episodes each:
      `n_action_steps` 4/8/16/32 → 6.0% / 52.0% / 56.0% / 36.0%. One inference-time flag recovered
      36%→56%; the curve has an interior optimum (4 collapses).
- [x] **E2 — training success rate was flat from 25k to 150k** while train loss fell 8x
      (24/24/22/22/32% at 25k/50k/100k/150k/200k, 50 eps each; measured run-to-run noise ±5pp).
      With no image augmentation and no held-out loss, train loss was not a proxy for rollout success.
- [x] **E4 — crop + GroupNorm + from-scratch backbone + `n_action_steps=16`, bundled with
      `drop_n_last_frames` 7→47: catastrophic regression** (2–10% over four 50-episode evals), with a
      distinctive pile-up of 20/50 episodes in `max_reward` ∈ [0.90, 0.95).
- [x] **E4b — isolated the cause: `drop_n_last_frames=47`.** `train/gt_path_length` (a ground-truth data
      statistic) had jumped 1.916→2.419, proving the training window distribution changed. Re-running E4
      with `drop_n_last_frames=7` restored it to 1.916 and gave 52.0% @25k → 62.0% @50k (n=50, monotonic).
      `drop_n_last_frames = horizon - n_action_steps - n_obs_steps + 1` is a stale formula from the
      original 16/8/2 config — lerobot's own `DiffusionConfig` also still ships 7 with `horizon=64`.
      Pinned at 7 in `configuration_flow_matching.py` with the evidence recorded inline.
- [x] **Corrected result: 69.0% over 200 episodes** (95% CI [62.6, 75.4]) at **50k** steps, vs. the
      baseline's published 65.4% over 500 episodes at 200k steps — parity at 1/4 the training budget and
      1/10 the sampling steps, with ~1.7x faster rollouts.
- [x] Confirmed defaults folded into `configuration_flow_matching.py`: `n_action_steps=16`,
      `drop_n_last_frames=7`, `crop_shape=(84,84)`, `use_group_norm=True`,
      `pretrained_backbone_weights=None`.
- [x] **Equal-budget 200k-step run** (`outputs/train/m4b_full_fixed`, wandb `hmsg3mdx`, 4h20m, batch 64,
      `--env_eval_freq=25000 --save_freq=25000 --eval.n_episodes=50`). In-training eval curve:
      54/66/60/64/68/70/**74**/74% at 25k…200k. Best checkpoint by rollout = 175k, matching the
      baseline's own best-of-periodic-eval methodology (it ships its 175k checkpoint too).
- [x] **M4 gate met — equal-budget head-to-head:** FM 175k = **73.5%** (147/200, CI [67.4, 79.6])
      vs. baseline's published 65.4% (500 eps, CI [61.2, 69.6]); two-proportion z = 2.07, p = 0.038.
      Marginal edge, not decisive; vs. our own 50-ep local baseline eval (62.0%) it's n.s. (p = 0.11).
      Achieved with 1/10 the sampling steps per chunk and ~1.7x faster rollouts.
      wandb: [g1gfo89x](https://wandb.ai/ameya555-ieee/lerobot/runs/g1gfo89x) (175k, n=200),
      [fxj42ubi](https://wandb.ai/ameya555-ieee/lerobot/runs/fxj42ubi) (200k, n=200).
- [x] **Training past 50k is not resolvable.** 50k (annealed) 69.0%, 200k final 69.0%, 200k best (175k)
      73.5% — all n=200, no pairwise difference significant (p = 0.32). 4x the compute, no detectable
      gain. Longer training shifts the *shape* of the failures rather than the rate: the 50k checkpoint
      has the highest `avg_max_reward` (0.971) and only 4/200 hard failures but 51 near-misses in
      [0.95, 1.0); 175k has 147 clean successes but 14 hard failures and `avg_max_reward` 0.928.
- [x] Benchmarked `ode_solver` ∈ {euler, heun, rk4} at **matched NFE** on the 175k checkpoint (n=200).
      **euler dominates everywhere; default stays euler.** heun@NFE10 = 7.0% and rk4@NFE12 = 29.0% vs.
      euler@NFE10 = 73.5% (p < 1e-10). Verified this is NOT a solver bug: (a) on a synthetic smooth field
      both higher-order solvers are strictly more accurate than euler; (b) on the real policy heun converges
      to euler as steps grow (num_steps 5/20/40 → 7/53/71%, vs euler@40 76%). The learned velocity field is
      only reliable on the training manifold `x_t = t·noise+(1-t)·action`; higher-order solvers evaluate it
      at off-path predicted points where it's unreliable. Full table + mechanism in `comparison_pusht.md`.
      Evals under `outputs/eval/m4b_175k_*` / `outputs/eval/conv_*`, logged to wandb as
      `m4b_175k_{solver}_nfe{N}` / `conv_*`.
- [x] **E5 — off-path training jitter (`path_noise_std=0.05`) does NOT rescue higher-order solvers, and
      hurts euler.** 50k run `e5_path_noise_005` (wandb `ab2hnfq7`), NFE-matched solver sweep n=200.
      heun/rk4 stayed on the floor (heun@NFE10 4.0% vs euler-trained 7.0%, p=0.19; rk4@NFE12 23.0% vs
      29.0%, p=0.17 — all matched-NFE comparisons n.s.). euler-vs-euler control at matched 50k steps:
      E4b (std=0) 69.0% vs e5 (std=0.05) **52.0%**, p=0.0005 — the jitter cost ~17pp. Measured that 0.05
      is well-scaled (heun probes 0.028–0.047 off-path), so "too small" is ruled out; the single-sample
      target is biased off-path, blurring the field rather than teaching the true marginal velocity.
      `path_noise_std` stays 0.0. Full writeup in `comparison_pusht.md`.
- [ ] Re-test `num_inference_steps` and uniform-vs-Beta time sampling on the fixed config (the earlier
      10-vs-100 result was measured on the old, unaugmented checkpoint).

## Phase 4 — Generalization (stretch, after PushT is solid)

- [ ] Confirm a second lerobot env/dataset (e.g. ALOHA) works with only `--dataset.repo_id`/`--env.type`
      overrides — verify feature keys/shapes first (`grep -n "class AlohaEnv" -A 30` in lerobot's
      `envs/configs.py`) since this wasn't checked during the PushT-focused planning pass
- [ ] Note any PushT-specific leakage found and fix it
