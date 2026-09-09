#!/usr/bin/env bash
# Minimal eval of a pretrained lerobot policy checkpoint, with rollout videos.
#
# lerobot-eval renders up to 10 episodes to <output_dir>/videos/*.mp4 by default
# (see EvalPipelineConfig / lerobot_eval.py: max_episodes_rendered=10 unless
# --eval.recording=true), so no extra flags are needed to get visualizations.
#
# `lerobot/diffusion_pusht` predates lerobot's processor-pipeline format and
# can't be loaded directly on lerobot>=0.6 -- run
# `python scripts/prepare_diffusion_pusht_checkpoint.py` once first (see that
# script's docstring for why) to produce models/diffusion_pusht_local, which
# is this script's default POLICY_PATH.
#
# --eval.use_async_envs=false works around a separate bug: lerobot's async
# vector env spawns workers via a `forkserver` context that never imports
# gym_pusht, so parallel rollouts fail to register the environment.
#
# lerobot-eval has no wandb integration itself (only lerobot-train does), so
# results are optionally pushed afterward via scripts/log_eval_to_wandb.py.
# Set WANDB_ENABLE=true to turn this on; requires `wandb login` beforehand.
#
# Usage:
#   scripts/eval_pretrained.sh [POLICY_PATH] [ENV_TYPE] [N_EPISODES] [OUTPUT_DIR] [-- EXTRA_ARGS...]
#
# Examples:
#   scripts/eval_pretrained.sh
#   scripts/eval_pretrained.sh models/diffusion_pusht_local pusht 10 outputs/eval/diffusion_pusht
#   WANDB_ENABLE=true scripts/eval_pretrained.sh
#   scripts/eval_pretrained.sh outputs/train/m4_full2/checkpoints/last/pretrained_model pusht 50 \
#     outputs/eval/m4_flow_matching_ode100 -- --policy.num_inference_steps=100
#
# Anything after a literal `--` is passed through to lerobot-eval verbatim. This is how to override a
# saved checkpoint's policy config at eval time (e.g. --policy.num_inference_steps=N) without retraining --
# num_inference_steps only controls how many Euler-integration steps conditional_sample() takes at
# inference; it's not baked into the trained weights.

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

POLICY_PATH="${POSITIONAL[0]:-models/diffusion_pusht_local}"
ENV_TYPE="${POSITIONAL[1]:-pusht}"
N_EPISODES="${POSITIONAL[2]:-10}"
OUTPUT_DIR="${POSITIONAL[3]:-outputs/eval/$(basename "$POLICY_PATH")}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"

echo "Evaluating policy=${POLICY_PATH} env=${ENV_TYPE} n_episodes=${N_EPISODES}"
echo "Output dir: ${OUTPUT_DIR} (rollout videos land in ${OUTPUT_DIR}/videos/)"

lerobot-eval \
  --policy.path="${POLICY_PATH}" \
  --env.type="${ENV_TYPE}" \
  --eval.n_episodes="${N_EPISODES}" \
  --eval.batch_size="${N_EPISODES}" \
  --eval.use_async_envs=false \
  --output_dir="${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"

if [ "${WANDB_ENABLE}" = "true" ]; then
  python "$(dirname "${BASH_SOURCE[0]}")/log_eval_to_wandb.py" "${OUTPUT_DIR}" \
    --policy-path="${POLICY_PATH}" --env-type="${ENV_TYPE}"
fi
