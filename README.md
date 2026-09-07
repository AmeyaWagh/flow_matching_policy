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
