#!/usr/bin/env python
"""Log an `lerobot-eval` run's metrics + rollout videos to Weights & Biases.

`lerobot-eval` has no built-in wandb integration -- only `lerobot-train`'s
`TrainPipelineConfig` has a `wandb: WandBConfig` field, and lerobot's
`WandBLogger` (lerobot.common.wandb_utils) is constructed from a full
`TrainPipelineConfig`, so it isn't reusable standalone for eval-only runs.
This script reads the `eval_info.json` that `lerobot-eval` already writes and
logs it to a plain wandb run, mirroring lerobot's own logging convention
(metrics under an "eval/" prefix, videos via `wandb.Video`).

Usage:
    python scripts/log_eval_to_wandb.py OUTPUT_DIR [--project lerobot] [--name RUN_NAME]

Requires `wandb login` (or $WANDB_API_KEY) to already be configured.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    """Parse CLI args, then log an existing lerobot-eval output_dir's results to wandb."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Directory passed as --output_dir to lerobot-eval")
    parser.add_argument("--project", default="lerobot")
    parser.add_argument("--name", default=None, help="wandb run name (defaults to the output_dir's name)")
    parser.add_argument("--policy-path", default=None, help="Recorded in the run config for reference")
    parser.add_argument("--env-type", default=None, help="Recorded in the run config for reference")
    args = parser.parse_args()

    eval_info_path = args.output_dir / "eval_info.json"
    if not eval_info_path.exists():
        raise FileNotFoundError(
            f"{eval_info_path} not found. Run lerobot-eval with --output_dir={args.output_dir} first."
        )
    eval_info = json.loads(eval_info_path.read_text())
    overall = eval_info["overall"]

    import wandb

    run = wandb.init(
        project=args.project,
        name=args.name or args.output_dir.name,
        job_type="eval",
        config={
            "policy_path": args.policy_path,
            "env_type": args.env_type,
            "output_dir": str(args.output_dir),
        },
    )

    metrics = {
        f"eval/{k}": v
        for k, v in overall.items()
        if isinstance(v, (int, float, str)) and k not in ("video_paths", "predicted_video_paths")
    }
    wandb.log(metrics)

    fps = 10  # PushT's EnvConfig.fps; adjust if evaluating a different env
    for video_path in overall.get("video_paths", []):
        wandb.log({"eval/video": wandb.Video(video_path, fps=fps, format="mp4")})

    print(f"Logged to {run.get_url()}")
    wandb.finish()


if __name__ == "__main__":
    main()
