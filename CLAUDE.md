# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A Flow Matching robot policy (`lerobot_policy_flow_matching`) built as an **out-of-tree plugin** for
HuggingFace's [lerobot](https://github.com/huggingface/lerobot), starting with the PushT task and
benchmarked against the `lerobot/diffusion_pusht` reference checkpoint. It is not meant to be
PushT-specific long-term — the architecture is driven by `input_features`/`output_features`, same as
lerobot's own `DiffusionPolicy`.

**Current state:** the environment, tooling, and baseline-eval scripts are in place. The actual policy
(`configuration_flow_matching.py`, `modeling_flow_matching.py`, `processor_flow_matching.py`) has not been
implemented yet — `src/lerobot_policy_flow_matching/__init__.py` is currently just a placeholder docstring.
See "Implementation plan" below before implementing it — it's the full, previously-approved design; follow
it rather than re-deriving the approach from scratch.

## Commands

```bash
# Environment (uv-managed; uv venvs ship without pip, so use uv add/uv sync, not pip)
uv venv --python 3.12
uv sync                    # base deps only (torch+cu128, lerobot[pusht,training,diffusion], robometric-frame)
uv sync --extra dev        # + ruff, mypy, interrogate, pre-commit (NOT installed by plain `uv sync`)
source .venv/bin/activate  # or prefix commands with `uv run` instead of activating

# Lint / type-check / docstring coverage / tests
uv run pre-commit run --all-files   # ruff-check, ruff-format, mypy, interrogate (all four must pass)
uv run ruff check --fix .
uv run ruff format .
uv run mypy src scripts
uv run interrogate -v src scripts   # docstring coverage, fails under 80% (pyproject.toml [tool.interrogate])
uv run pytest                       # tests/ (empty so far — testpaths = ["tests"] in pyproject.toml)
uv run pytest tests/test_foo.py::test_bar   # single test, once tests exist

# Evaluate the pretrained diffusion_pusht baseline (see "Known workarounds" below for why the prep step exists)
python scripts/prepare_diffusion_pusht_checkpoint.py         # one-time: writes models/diffusion_pusht_local/
scripts/eval_pretrained.sh                                   # lerobot-eval + save rollout videos
scripts/eval_pretrained.sh POLICY_PATH ENV_TYPE N_EPISODES OUTPUT_DIR
WANDB_ENABLE=true scripts/eval_pretrained.sh                  # also push eval_info.json + videos to wandb

# Training / eval a policy directly via lerobot's own CLIs (once the plugin is implemented)
lerobot-train --policy.type=flow_matching --dataset.repo_id=lerobot/pusht --env.type=pusht ...
lerobot-eval --policy.path=<checkpoint_dir_or_hub_id> --env.type=pusht --eval.use_async_envs=false ...
```

## Implementation plan

The full design (package layout, exact config field list to keep/drop/add vs. `DiffusionConfig`, model
reuse strategy, `compute_loss`/sampling code sketches, processor design, and generalization argument) lives
in **`docs/plan.md`** — read it before implementing any of `configuration_flow_matching.py`,
`modeling_flow_matching.py`, or `processor_flow_matching.py`; it's the previously-approved design, not a
draft to re-derive.

**`docs/checklist.md`** tracks granular progress against that plan (per-file, per-test, per-milestone
checkboxes) — update it as work lands rather than letting it drift from reality.

Milestone gates (detailed commands in `docs/plan.md`): **M1** registration + 1-step training smoke test →
**M2** short training run with visibly decreasing loss → **M3** nonzero rollout success rate on PushT →
**M4** full training run + side-by-side comparison against `lerobot/diffusion_pusht`, written up in
`docs/comparison_pusht.md`.

## Known workarounds (all deliberate, not accidental — don't revert without re-checking the underlying issue)

- **`lerobot/diffusion_pusht` needs local prep before it can be loaded** (lerobot ≥0.6): the checkpoint
  predates lerobot's processor-pipeline normalization format, and lerobot's own migration script
  (`lerobot.processor.migrate_policy_normalization`) hits a draccus serialization bug re-saving this
  checkpoint's config (`crop_shape` tuple fails to encode). `scripts/prepare_diffusion_pusht_checkpoint.py`
  works around this by taking only the migration script's *processor* output
  (`policy_preprocessor.json`/`policy_postprocessor.json` + normalizer safetensors, which save fine) and
  pairing it with the checkpoint's original, untouched `config.json`/`model.safetensors`.
- **`--eval.use_async_envs=false` is required** when running `lerobot-eval`/`lerobot-train` against `pusht`:
  lerobot's async vector env spawns workers via a `forkserver` context that never imports `gym_pusht`, so
  parallel rollouts fail with `NamespaceNotFound`. Sync envs run in-process and avoid it.
  `scripts/eval_pretrained.sh` already sets this.
- **`lerobot-eval` has no wandb integration** — only `lerobot-train`'s `TrainPipelineConfig` has a `wandb`
  field, and lerobot's `WandBLogger` requires a full `TrainPipelineConfig` to construct, so it isn't
  reusable standalone. `scripts/log_eval_to_wandb.py` reads the `eval_info.json` that `lerobot-eval` already
  writes and logs it to a plain wandb run instead (metrics under an `eval/` prefix, matching lerobot's own
  convention).
- **pre-commit's `mypy` hook runs in its own isolated venv** — any third-party package imported in
  type-checked code needs listing in that hook's `additional_dependencies` (see `wandb` in
  `.pre-commit-config.yaml`), or mypy can't see past the import and misreports real attributes as undefined.

## Dependency notes

- Python ≥3.12; PyTorch/torchvision pulled from a pinned `pytorch-cu128` uv index (mirrors lerobot's own
  pyproject.toml) for consistent CUDA wheels on Linux.
- `lerobot[pusht,training,diffusion]` is pinned to `>=0.6.1,<0.7`. The `diffusion` extra (`diffusers`) is
  pulled in only so the *reference* `diffusion_pusht` checkpoint can be loaded for comparison — the Flow
  Matching policy itself should never import `diffusers`.
- `robometric-frame` is a general dependency (unrelated to lerobot).
