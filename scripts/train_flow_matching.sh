#!/usr/bin/env bash
# Train the Flow Matching policy via lerobot-train.
#
# Wraps `lerobot-train --policy.type=flow_matching` with the workarounds this repo already needs
# (see CLAUDE.md "Known workarounds"):
#   --eval.use_async_envs=false  lerobot's async vector env spawns workers via a `forkserver` context
#                                 that never imports `gym_pusht`, so periodic in-training eval rollouts
#                                 would fail to register the environment.
#   --policy.push_to_hub=false   no Hub repo is configured for local dev runs; lerobot-train otherwise
#                                 refuses to start (`'repo_id' argument missing`).
#
# Usage:
#   scripts/train_flow_matching.sh [STEPS] [BATCH_SIZE] [OUTPUT_DIR] [DATASET_REPO_ID] [ENV_TYPE] [-- EXTRA_ARGS...]
#
# Examples:
#   scripts/train_flow_matching.sh                                     # full run: 200000 steps
#   scripts/train_flow_matching.sh 5000 64 outputs/train/m2_short       # M2 short run
#   WANDB_ENABLE=true scripts/train_flow_matching.sh 5000
#   scripts/train_flow_matching.sh 30000 64 outputs/train/m3 -- --env_eval_freq=0   # disable periodic in-training eval
#
# Anything after a literal `--` is passed through to lerobot-train verbatim (e.g. to override
# --policy.time_embed_scale, --save_freq, ...).
#
# To disable lerobot's periodic in-training rollout eval, use `--env_eval_freq=0` -- NOT
# `--eval.n_episodes=0`. The latter looks like it should work but doesn't: EvalConfig.__post_init__
# auto-computes `eval.batch_size` as `min(cpu_based_default, n_episodes, 64)`, so n_episodes=0 collapses
# batch_size to 0 too, and the periodic eval (still scheduled at every `env_eval_freq`, default 20000
# steps) then crashes with `ValueError: n_envs must be at least 1` when it fires -- discovered the hard
# way after a 20000-step run hit exactly that wall. The checkpoint at that step is unaffected (saved
# just before the crash), but the crash still kills the process, so 10000 more steps get lost if the
# target was e.g. 30000. `--env_eval_freq=0` disables the periodic eval outright, cleanly.

set -euo pipefail

POSITIONAL=()
EXTRA_ARGS=()
FOUND_SEP=false
for arg in "$@"; do
  if [ "$arg" = "--" ]; then
    FOUND_SEP=true
    continue
  fi
  if [ "$FOUND_SEP" = true ]; then
    EXTRA_ARGS+=("$arg")
  else
    POSITIONAL+=("$arg")
  fi
done

STEPS="${POSITIONAL[0]:-200000}"
BATCH_SIZE="${POSITIONAL[1]:-64}"
OUTPUT_DIR="${POSITIONAL[2]:-outputs/train/flow_matching_pusht}"
DATASET_REPO_ID="${POSITIONAL[3]:-lerobot/pusht}"
ENV_TYPE="${POSITIONAL[4]:-pusht}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"

echo "Training flow_matching on ${DATASET_REPO_ID} (env=${ENV_TYPE}): steps=${STEPS} batch_size=${BATCH_SIZE}"
echo "Output dir: ${OUTPUT_DIR}"

lerobot-train \
  --policy.type=flow_matching \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --env.type="${ENV_TYPE}" \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --eval.use_async_envs=false \
  --policy.push_to_hub=false \
  --wandb.enable="${WANDB_ENABLE}" \
  --output_dir="${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"
