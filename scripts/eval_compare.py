#!/usr/bin/env python
"""Diff two `lerobot-eval` runs side by side, from local output dirs and/or wandb runs.

Not a new eval harness -- both runs already went through the identical `eval_policy_all`; this just
loads their `overall`/`eval/*` metrics and prints a comparison table. Each argument can be:
  - a local `lerobot-eval --output_dir=...` directory (reads `eval_info.json`), or
  - a wandb run, as a URL (`https://wandb.ai/<entity>/<project>/runs/<run_id>`) or short form
    (`<entity>/<project>/<run_id>`) -- requires the run to have been logged via
    `scripts/log_eval_to_wandb.py` (or `WANDB_ENABLE=true scripts/eval_pretrained.sh`), and a
    configured `WANDB_API_KEY` to fetch it.

Usage:
    python scripts/eval_compare.py SOURCE_A SOURCE_B [--label-a NAME] [--label-b NAME]
"""

import argparse
import json
import re
from pathlib import Path

METRICS = ["pc_success", "avg_max_reward", "avg_sum_reward", "n_episodes", "eval_ep_s"]

WANDB_URL_RE = re.compile(r"https?://wandb\.ai/([^/]+)/([^/]+)/runs/([^/?]+)")


def load_overall(source: str) -> dict:
    """Load eval metrics from a local `lerobot-eval` output_dir
    or a wandb run (URL or entity/project/run_id)."""
    eval_info_path = Path(source) / "eval_info.json"
    if eval_info_path.exists():
        return json.loads(eval_info_path.read_text())["overall"]

    url_match = WANDB_URL_RE.match(source)
    if url_match:
        run_path = "/".join(url_match.groups())
    elif source.count("/") == 2:
        run_path = source
    else:
        raise FileNotFoundError(
            f"{eval_info_path} not found, and '{source}' doesn't look like a wandb run "
            "(expected a https://wandb.ai/... URL or 'entity/project/run_id')."
        )

    import wandb  # local import: only needed for the wandb-run code path

    summary = wandb.Api().run(run_path).summary
    return {key.removeprefix("eval/"): value for key, value in summary.items() if key.startswith("eval/")}


def default_label(source: str) -> str:
    """A short display label for a source: the directory name for local runs, the source itself otherwise."""
    local_dir = Path(source)
    return local_dir.name if (local_dir / "eval_info.json").exists() else source


def fmt(value: object) -> str:
    """Render a metric value for the comparison table, rounding floats for readability."""
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def main() -> None:
    """Parse CLI args and print a side-by-side comparison table for two eval runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_a", help="Local output_dir, or wandb run URL / entity/project/run_id")
    parser.add_argument("source_b", help="Local output_dir, or wandb run URL / entity/project/run_id")
    parser.add_argument("--label-a", default=None)
    parser.add_argument("--label-b", default=None)
    args = parser.parse_args()

    label_a = args.label_a or default_label(args.source_a)
    label_b = args.label_b or default_label(args.source_b)
    overall_a = load_overall(args.source_a)
    overall_b = load_overall(args.source_b)

    name_width = max(len(m) for m in METRICS)
    col_width = max(len(label_a), len(label_b), 10)
    header = f"{'metric':<{name_width}}  {label_a:>{col_width}}  {label_b:>{col_width}}"
    print(header)
    print("-" * len(header))
    for metric in METRICS:
        val_a = overall_a.get(metric, "n/a")
        val_b = overall_b.get(metric, "n/a")
        if metric != "n_episodes":
            # Normalize int-vs-float across sources (wandb summaries store e.g. pc_success as an int;
            # local eval_info.json stores it as a float) so the two columns format consistently.
            val_a = float(val_a) if isinstance(val_a, (int, float)) else val_a
            val_b = float(val_b) if isinstance(val_b, (int, float)) else val_b
        print(f"{metric:<{name_width}}  {fmt(val_a):>{col_width}}  {fmt(val_b):>{col_width}}")


if __name__ == "__main__":
    main()
