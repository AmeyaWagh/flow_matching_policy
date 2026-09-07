# flow_matching_policy
Flow Matching For Robot Policy

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and Python >=3.12.

```bash
# Create the virtual environment (once)
uv venv --python 3.12

# Install/sync dependencies (torch, lerobot[pusht,training,diffusion], this package, etc.)
uv sync

# Activate the environment
source .venv/bin/activate
```

Once activated, `python`, `lerobot-train`, and `lerobot-eval` resolve to the venv's versions. To leave the environment, run `deactivate`.

Alternatively, skip activation and prefix one-off commands with `uv run` (e.g. `uv run lerobot-train ...`), which resolves against `.venv` automatically.

Verify the setup:

```bash
python -c "import torch, lerobot; print(torch.__version__, torch.cuda.is_available(), lerobot.__version__)"
```

## Evaluate the pretrained diffusion_pusht baseline

`lerobot/diffusion_pusht` predates lerobot's processor-pipeline normalization format, so it needs a
one-time local prep step before `lerobot-eval` can load it (see the script's docstring for why):

```bash
python scripts/prepare_diffusion_pusht_checkpoint.py   # writes models/diffusion_pusht_local/
scripts/eval_pretrained.sh                             # eval + save rollout videos
```

`scripts/eval_pretrained.sh [POLICY_PATH] [ENV_TYPE] [N_EPISODES] [OUTPUT_DIR]` wraps `lerobot-eval` and
defaults to evaluating `models/diffusion_pusht_local` on `pusht` for 10 episodes. Rollout videos are
saved to `<output_dir>/videos/` automatically (up to 10 episodes) — no extra flags needed.
