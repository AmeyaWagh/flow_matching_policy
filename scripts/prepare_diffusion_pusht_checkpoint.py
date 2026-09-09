#!/usr/bin/env python
"""Prepare a locally-loadable copy of `lerobot/diffusion_pusht` for lerobot>=0.6.

The checkpoint predates lerobot's processor-pipeline normalization format, so
`lerobot-eval`/`lerobot-train` refuse to load it directly (missing
`policy_preprocessor.json`/`policy_postprocessor.json` on the Hub repo).

`lerobot.processor.migrate_policy_normalization` is meant to fix this, but as of
lerobot 0.6.1 it hits a draccus serialization bug while re-saving this specific
checkpoint's config (a `crop_shape` tuple fails to encode), so the model weights
and config it produces are corrupted/incomplete. Its *processor* output
(`policy_preprocessor.json` / `policy_postprocessor.json` + normalizer
safetensors) is fine on its own, since that doesn't depend on the buggy
re-save step.

This script runs the migration to get those processor files, then assembles a
working local checkpoint directory by pairing them with the *original*
`config.json`/`model.safetensors` (which load fine as-is; only the processor
metadata was ever missing).

Usage:
    python scripts/prepare_diffusion_pusht_checkpoint.py [output_dir]
"""

import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "lerobot/diffusion_pusht"


def main() -> None:
    """Assemble a locally-loadable diffusion_pusht checkpoint at the given output directory."""
    output_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "models/diffusion_pusht_local")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading original checkpoint files for {REPO_ID}...")
    snapshot_dir = Path(snapshot_download(REPO_ID, allow_patterns=["config.json", "model.safetensors"]))
    shutil.copyfile(snapshot_dir / "config.json", output_dir / "config.json")
    shutil.copyfile(snapshot_dir / "model.safetensors", output_dir / "model.safetensors")

    with tempfile.TemporaryDirectory() as tmp:
        migrated_dir = Path(tmp) / "migrated"
        print("Running migrate_policy_normalization (expected to warn/crash on the config re-save)...")
        import subprocess

        subprocess.run(
            [
                sys.executable,
                "-m",
                "lerobot.processor.migrate_policy_normalization",
                "--pretrained-path",
                REPO_ID,
                "--output-dir",
                str(migrated_dir),
            ],
            check=False,  # known to exit non-zero on the config re-save bug; we only need the processor files
        )

        for name in (
            "policy_preprocessor.json",
            "policy_postprocessor.json",
            "policy_preprocessor_step_3_normalizer_processor.safetensors",
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        ):
            src = migrated_dir / name
            if not src.exists():
                raise FileNotFoundError(
                    f"Expected migration output '{name}' not found in {migrated_dir}. "
                    "The migration script's processor-saving step may have changed; "
                    "inspect its output manually."
                )
            shutil.copyfile(src, output_dir / name)

    print(f"Ready: {output_dir}")
    print(f"  lerobot-eval --policy.path={output_dir} --env.type=pusht --eval.use_async_envs=false ...")


if __name__ == "__main__":
    main()
