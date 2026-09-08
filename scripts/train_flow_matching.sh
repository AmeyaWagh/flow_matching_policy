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
#   scripts/train_flow_matching.sh 5000 64 outputs/train/m2_short -- --eval.n_episodes=0
#
# Anything after a literal `--` is passed through to lerobot-train verbatim (e.g. to override
# --policy.time_embed_scale, --eval.n_episodes, --save_freq, ...).

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
WANDB_ENABLE="${WANDB_ENABLE:-false}"

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
