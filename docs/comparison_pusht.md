# Flow Matching vs. Diffusion Policy — PushT comparison (M4)

## Headline result

| | Flow Matching (ours) | `lerobot/diffusion_pusht` |
|---|---:|---:|
| Training steps | **50,000** | 200,000 (shipped checkpoint is the 175k one, selected by periodic eval) |
| Inference steps per action chunk | **10** (forward-Euler ODE) | ~100 (DDPM) |
| `pc_success` | **69.0%** (138/200 episodes, 95% CI [62.6, 75.4]) | 65.4% (500 episodes, HF model card) / 62.0% (50 episodes, our local eval) |
| `avg_max_reward` | 0.971 | 0.955 (model card) / 0.965 (local) |
| Eval wall-clock per episode | **0.65 s** | 1.12 s |

At 1/4 of the baseline's training budget and 1/10 of its sampling steps, the Flow Matching policy matches
the diffusion baseline's published success rate (the baseline's 65.4% sits inside our 95% CI). Reproduce:

```bash
lerobot-eval --policy.path=outputs/train/e4b_dropnlf7/checkpoints/050000/pretrained_model \
  --env.type=pusht --eval.n_episodes=200 --eval.batch_size=50 --eval.use_async_envs=false \
  --output_dir=outputs/eval/e4b_step050000_nas16_n200
```
wandb: [e4b_step050000_nas16_n200](https://wandb.ai/ameya555-ieee/lerobot/runs/113nyoss) (eval),
[mwwcfwtk](https://wandb.ai/ameya555-ieee/lerobot/runs/mwwcfwtk) (training).

**Caveat, stated up front:** this is not yet an equal-budget head-to-head. Ours is 50k steps with a
fully-annealed cosine schedule; the baseline is 200k. lerobot's `CosineDecayWithWarmupSchedulerConfig.build()`
auto-scales `num_decay_steps` down to the actual run length when `steps < num_decay_steps`, so a 50k run is a
*complete* run, not a truncated prefix of a 200k one — the two are not directly comparable step-for-step, and
a 200k Flow Matching run is still outstanding. See "Open items".

## How we got here: the original M4 comparison was wrong in two ways

The first M4 comparison reported Flow Matching at 32.0% vs. diffusion at 62.0% and concluded "correct but
less sample-efficient." Both halves of that conclusion turned out to be artifacts of configuration, not of
the generative process. Two separate causes, found in order:

### 1. The two policies were not architecture-matched (E1)

The first comparison claimed "the one deliberate difference between the two policies is the generative
process." That was false. `FlowMatchingConfig` faithfully mirrored lerobot 0.6's *current* `DiffusionConfig`
defaults, but the `lerobot/diffusion_pusht` checkpoint was trained under the *older* ones:

| | diffusion_pusht (baseline) | FM, original M4 run |
|---|---:|---:|
| `horizon` | 16 | 64 |
| `n_action_steps` (open-loop execution) | 8 | **32** |
| `crop_shape` | [84, 84] random crop | **None** (no augmentation at all) |
| `use_group_norm` / `pretrained_backbone_weights` | True / None | **False / ImageNet** |

Sweeping `n_action_steps` at eval time on the *same, unmodified* M4 weights (50 episodes each):

| `n_action_steps` | 4 | 8 | 16 | 32 (as evaluated originally) |
|---|---:|---:|---:|---:|
| `pc_success` | 6.0% | 52.0% | **56.0%** | 36.0% |
| `avg_max_reward` | 0.595 | 0.883 | 0.899 | 0.793 |

wandb: [nas4](https://wandb.ai/ameya555-ieee/lerobot/runs/04pozifd),
[nas8](https://wandb.ai/ameya555-ieee/lerobot/runs/1a2ehh8r),
[nas16](https://wandb.ai/ameya555-ieee/lerobot/runs/paxmis34),
[nas32](https://wandb.ai/ameya555-ieee/lerobot/runs/tdk9kxbz).

A single inference-time flag moved success from 36% to 56% — so most of the original "FM is half as good"
gap was an execution-horizon mismatch. The curve has an interior optimum: `n_action_steps=4` *collapses* to
6% despite the highest `avg_sum_reward` of the sweep, the signature of replanning so often that the policy
chatters between modes (each replan draws an independent noise sample). 8 and 16 are statistically tied at
n=50; 16 is now the default.

### 2. Training was not actually converging — and then a config "fix" made it much worse (E2, E4, E4b)

Evaluating the original run's saved checkpoints (50 episodes each, `n_action_steps=32`) showed the success
rate was **flat** while the training loss fell 8x:

| steps | 25k | 50k | 100k | 150k | 200k |
|---|---:|---:|---:|---:|---:|
| `pc_success` | 24.0% | 24.0% | 22.0% | 22.0% | 32.0% |
| train loss | 0.0111 | 0.0070 | 0.0042 | 0.0024 | 0.0014 |

(wandb: [lf3916bs](https://wandb.ai/ameya555-ieee/lerobot/runs/lf3916bs),
[m4spxzv4](https://wandb.ai/ameya555-ieee/lerobot/runs/m4spxzv4),
[nltscnyx](https://wandb.ai/ameya555-ieee/lerobot/runs/nltscnyx),
[gvhbvzwi](https://wandb.ai/ameya555-ieee/lerobot/runs/gvhbvzwi),
[r8ejg326](https://wandb.ai/ameya555-ieee/lerobot/runs/r8ejg326).) Run-to-run noise at n=50 is about ±5pp —
measured by repeating one identical eval and getting 52% then 58% — so essentially all of that variation is
noise. **175k of the 200k steps bought nothing measurable**: with no image augmentation and no held-out loss,
train loss was not a usable proxy for rollout success.

Fixing that (E4) — `crop_shape=(84,84)` random crop, `use_group_norm=True`,
`pretrained_backbone_weights=None`, `n_action_steps=16` — while *also* changing
`drop_n_last_frames` from 7 to 47 produced a **catastrophic regression**:

| E4 checkpoint / `n_action_steps` | 25k / 16 | 25k / 8 | 50k / 16 | 50k / 8 |
|---|---:|---:|---:|---:|
| `pc_success` | 10.0% | 8.0% | **2.0%** | 6.0% |
| `avg_max_reward` | 0.891 | 0.793 | 0.855 | 0.772 |

The failure had a distinctive signature: **20 of 50 episodes piled into `max_reward` ∈ [0.90, 0.95)** — a
wall just below the success threshold — and the run posted the *highest* `avg_sum_reward` ever recorded here
(141). (In lerobot's PushT, success ⟺ `max_reward == 1.0` exactly; verified across 150 episodes.) The policy
parked the block near-aligned for the whole episode and never closed the last few percent of coverage.

The cause was visible in the *training* logs, not the eval: `train/gt_path_length` — a ground-truth data
statistic — jumped from 1.916 to **2.419** (+26%) between the two runs. Only `drop_n_last_frames` can change
the training window distribution like that. With 47, ~40% of each ~125-frame PushT episode is removed from
`EpisodeAwareSampler`'s start indices, so the terminal fine-alignment phase is only ever seen at chunk
positions 17–64 and never at 0–15 — the only slice receding-horizon execution ever runs.

Ablation (E4b): identical to E4 but `drop_n_last_frames=7`. `train/gt_path_length` returned to 1.916, and
in-training rollout eval (n=50) went **52.0% @ 25k → 62.0% @ 50k**, monotonic, versus E4's 10% → 2%
collapse. The 200-episode eval of that checkpoint is the 69.0% headline above.

**`drop_n_last_frames = horizon - n_action_steps - n_obs_steps + 1` is wrong.** That formula comes from the
original diffusion-policy config (16/8/2 → 7) and does not generalize to `horizon=64`. lerobot's own
`DiffusionConfig` also still ships 7 alongside `horizon=64`/`n_action_steps=32`; the comment claiming the
formula is stale upstream too. The value is now pinned at 7 with this reasoning recorded in
`configuration_flow_matching.py`.

## Confirmed configuration

`n_obs_steps=2`, `horizon=64`, `n_action_steps=16`, `drop_n_last_frames=7`, `crop_shape=(84,84)` with
`crop_is_random=True`, `use_group_norm=True`, `pretrained_backbone_weights=None`, `down_dims=(512,1024,2048)`,
`num_inference_steps=10`, `time_embed_scale=1000.0`, `time_sampling_alpha/beta=1.5/1.0`, batch 64, AdamW
preset with cosine decay + 500 warmup steps. Train with:

```bash
scripts/train_flow_matching.sh 50000 64 outputs/train/fm_pusht -- \
  --env_eval_freq=25000 --save_freq=25000 --eval.n_episodes=50
```

## Inference-step count: more is not better

Tested on the original M4 weights: `num_inference_steps=100` scored **24.0%** vs **32.0%** at the default 10
(wandb [i1d28jkd](https://wandb.ai/ameya555-ieee/lerobot/runs/i1d28jkd)) — 10x more Euler steps made it
worse. The likely mechanism is the training-time sampler: `t ~ Beta(1.5, 1.0)` has CDF `t^1.5`, so
P(t<0.1) = 3.2% and P(t<0.01) = 0.1%. lerobot's `euler_integrate` steps `time = 1 + step·dt`, so with 10
steps the lowest velocity query is at `t=0.1` — exactly where training data ends — while 100 steps spends
10% of its compute at `t < 0.1`, a region the model barely saw. This has not been independently confirmed
(one data point, and it predates the config fixes); it is the leading hypothesis, not an established fact.

## Open items

- **Full 200k-step run for an equal-budget head-to-head.** The current 69% comes from a 50k run; the
  baseline is 200k. Because the LR schedule auto-scales, a 200k run is a genuinely different run, not an
  extension, so the result cannot be extrapolated. Recipe to match the baseline: batch 64,
  `--env_eval_freq=25000 --save_freq=25000 --eval.n_episodes=50`, best-checkpoint selection by rollout
  (the baseline itself ships its 175k checkpoint, not its 200k one). ~4h20m on an RTX 4090.
- **Remaining headroom is in the near-miss band.** In the 200-episode eval, 51 episodes landed in
  `max_reward` ∈ [0.95, 1.0) — near-misses, not failures (only 7 episodes scored below 0.8). Closing those
  is worth more than anything else on this list.
- **Re-test `num_inference_steps`** (and uniform vs. Beta time sampling) on the fixed config; the 10-vs-100
  result above was measured on the old, broken-augmentation checkpoint.
- **`time_embed_scale` has still never been swept** — the default worked on the first try at every
  milestone, which is not the same as tuned.

## Conclusion

The Flow Matching policy matches the `lerobot/diffusion_pusht` baseline's published PushT success rate
(69.0% ± 6.4 over 200 episodes vs. 65.4% over 500) at a quarter of the training steps and a tenth of the
sampling steps per action chunk, with ~1.7x faster rollouts. The architecture is unchanged from
`DiffusionPolicy` apart from the generative process; every gap found between the two policies during M4
traced to configuration (execution horizon, image augmentation, and the `drop_n_last_frames` sampler bug),
not to flow matching itself. An equal-budget 200k-step comparison is still outstanding.
