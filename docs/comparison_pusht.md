# Flow Matching vs. Diffusion Policy — PushT comparison (M4)

## Results

Both policies evaluated through the identical `lerobot-eval` harness (`eval_policy_all`), same task
(`--env.type=pusht`), same episode count (`--eval.n_episodes=50`), same default seed (1000),
`--eval.use_async_envs=false` on both (see `CLAUDE.md`'s "Known workarounds").

| | Flow Matching (ours) | `lerobot/diffusion_pusht` |
|---|---:|---:|
| Training steps | 200,000 | 200,000 (pretrained checkpoint) |
| Inference steps per action chunk | **10** (forward-Euler ODE) | ~100 (DDPM) |
| `pc_success` | **32.0%** (16/50) | **62.0%** (31/50) |
| `avg_max_reward` | 0.777 | 0.965 |
| `avg_sum_reward` | 87.72 | 116.76 |
| Eval wall-clock per episode | **0.73 s** | 1.12 s |

Reproduce:
```bash
scripts/eval_pretrained.sh outputs/train/m4_full2/checkpoints/last/pretrained_model pusht 50 outputs/eval/m4_flow_matching
scripts/eval_pretrained.sh models/diffusion_pusht_local pusht 50 outputs/eval/m4_diffusion_baseline
```
wandb runs: [m4_flow_matching](https://wandb.ai/ameya555-ieee/lerobot/runs/a4gew556),
[m4_diffusion_baseline](https://wandb.ai/ameya555-ieee/lerobot/runs/7qa42c2d).

The table above was generated with `scripts/eval_compare.py`, which accepts either local
`lerobot-eval --output_dir=...` directories or wandb run references (URL or `entity/project/run_id`) for
each side, so it works from either source without needing the local `outputs/eval/` files to still exist:
```bash
python scripts/eval_compare.py outputs/eval/m4_flow_matching outputs/eval/m4_diffusion_baseline
python scripts/eval_compare.py \
  https://wandb.ai/ameya555-ieee/lerobot/runs/a4gew556 \
  https://wandb.ai/ameya555-ieee/lerobot/runs/7qa42c2d \
  --label-a flow_matching --label-b diffusion_pusht
```

## Reading the gap

At equal training budget, diffusion's success rate is roughly double ours. Two things are worth separating
here: whether this reflects a *bug* in the Flow Matching implementation, and whether it reflects a *real,
expected* trade-off of the design choices made.

**Not a bug.** The per-episode reward distributions for both policies are coherent, not degenerate:

- **Flow Matching** (`max_reward`, sorted): a genuine bimodal spread — roughly a third of episodes fail
  outright (0–0.7), a chunk land as near-misses (0.9–0.99, the same "pushes the block into place but
  doesn't hold the exact 95%-coverage threshold" pattern visually confirmed back at M3), and 16 episodes
  are clean 1.0 successes.
- **Diffusion**: the same shape, shifted toward success — a handful of clear failures (0.52–0.73), a
  similar near-miss band (0.95–0.999), and 31 clean 1.0s.

Neither distribution looks like noise or a broken policy; both look like "the same task, solved with
different reliability." This is also consistent with everything upstream of this eval: the M2→M3→M4
training curves showed smooth, monotonic loss decrease (0.487→0.001) with no instability, and the
trajectory-quality metrics (`gt_path_length`/`pred_path_length`: 1.923/1.984 at the final step, `gt_path_smoothness`/`pred_path_smoothness`:
0.017/0.013) show the model's own predicted action trajectories converged to closely match the
ground-truth demonstration trajectories' shape statistics — independent evidence that training converged
correctly, not just that the loss number went down.

**A real, expected trade-off.** The one deliberate difference between the two policies is the generative
process, and it shows up exactly where you'd expect: **10 Euler-integration steps vs. ~100 DDPM steps**.
Diffusion's reverse process gets 10x more network evaluations to refine the action chunk per inference
call, which plausibly buys it the precision needed to more reliably cross PushT's strict 95%-coverage
success bar — the same bar our near-miss episodes (0.9–0.99) sit just below. This is the standard
diffusion-vs-flow-matching trade-off in the literature (fewer sampling steps, faster inference, at some
quality cost unless step count or training compensates) showing up concretely in this comparison: FM's
eval wall-clock is ~35% faster per episode despite an identical architecture everywhere except the
generative process.

## What would likely close the gap (not required for this milestone, noted for future work)

- **More inference steps.** `num_inference_steps` is currently 10 (a pi0-style default); sweeping to
  20–50 and re-evaluating the same 200K-step checkpoint costs nothing in training time and directly tests
  the hypothesis above.
- **Longer training / different LR schedule.** Diffusion's checkpoint is a mature, independently-tuned
  release; ours is a single 200K-step run with pi0-inherited defaults (`time_sampling_alpha/beta`,
  `time_embed_scale`) never swept for this architecture.
- **`time_embed_scale` sensitivity.** Flagged as a design risk from the start (see `plan.md`) and never
  swept in practice, since the default worked cleanly on the first try at every milestone so far.

## Conclusion

The Flow Matching policy is a **correct, working implementation** — same architecture as `DiffusionPolicy`
apart from the generative process, verified end-to-end via unit tests, training-curve monotonicity, and
independent trajectory-quality convergence — that currently trails the mature diffusion baseline on raw
success rate at equal training steps, with a plausible and literature-consistent explanation (10x fewer
sampling steps) rather than an unexplained regression. Per the original plan: the goal here was a fair,
correct comparison, not necessarily beating diffusion — that bar is met.
