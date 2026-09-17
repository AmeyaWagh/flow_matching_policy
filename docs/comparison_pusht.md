# Flow Matching vs. Diffusion Policy — PushT comparison (M4)

## Headline result — equal-budget head-to-head (200k vs. 200k)

| | Flow Matching (ours) | `lerobot/diffusion_pusht` |
|---|---:|---:|
| Training steps | 200,000 | 200,000 |
| Checkpoint used | **175k**, best of 8 periodic evals | **175k**, best of periodic evals (their published pick) |
| Inference steps per action chunk | **10** (forward-Euler ODE) | ~100 (DDPM) |
| `pc_success` | **73.5%** (147/200 eps), 95% CI **[67.4, 79.6]** | 65.4% (500 eps, HF model card), 95% CI [61.2, 69.6] |
| `avg_max_reward` | 0.928 | 0.955 (model card) / 0.965 (our local 50-ep eval) |
| Eval wall-clock per episode | **0.66 s** | 1.12 s |

**At matched training budget and matched checkpoint-selection methodology, the Flow Matching policy edges out
the diffusion baseline** — 73.5% vs. 65.4%, a two-proportion z-test gives z = 2.07, **p = 0.038**. That is a
marginal result, not a decisive one: it clears p<0.05 but would not survive multiple-comparison correction,
and against our own local 50-episode baseline eval (62.0%, CI [48.5, 75.5]) the difference is not significant
(p = 0.11) because n=50 is simply too small. The defensible claim is **parity or a modest edge**, achieved
with 1/10 the sampling steps per action chunk and ~1.7x faster rollouts.

```bash
lerobot-eval --policy.path=outputs/train/m4b_full_fixed/checkpoints/175000/pretrained_model \
  --env.type=pusht --eval.n_episodes=200 --eval.batch_size=50 --eval.use_async_envs=false \
  --output_dir=outputs/eval/m4b_step175000_n200
```
wandb: [m4b_step175000_n200](https://wandb.ai/ameya555-ieee/lerobot/runs/g1gfo89x) (eval),
[hmsg3mdx](https://wandb.ai/ameya555-ieee/lerobot/runs/hmsg3mdx) (training, 4h20m).

### Training longer than 50k bought essentially nothing

The 200k run's own in-training periodic evals (50 episodes each, `n_action_steps=16`):

| steps | 25k | 50k | 75k | 100k | 125k | 150k | **175k** | 200k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pc_success` | 54.0% | 66.0% | 60.0% | 64.0% | 68.0% | 70.0% | **74.0%** | 74.0% |
| train loss | 0.0147 | 0.0100 | 0.0075 | 0.0056 | 0.0041 | 0.0030 | 0.0024 | 0.0020 |

The trend is upward but noisy (±~6.5pp SE at n=50), and the 175k/200k tie at the n=50 resolution is why both
were re-evaluated at n=200. Tightened numbers:

| checkpoint | `pc_success` (n=200) | 95% CI |
|---|---:|---|
| m4b 175k | **73.5%** | [67.4, 79.6] |
| m4b 200k | 69.0% | [62.6, 75.4] |
| E4b 50k (separate, fully-annealed 50k run) | 69.0% | [62.6, 75.4] |

**None of these three differ significantly** (175k vs. 200k: p = 0.32; 175k vs. the 50k run: p = 0.32). Stated
plainly: **4x the training compute (50k → 200k) produced no statistically resolvable improvement.** A fully
annealed 50k run reaches 69.0% in ~1h05m; the 200k run reaches 73.5% at its best checkpoint in 4h20m. Once the
`drop_n_last_frames` bug was fixed, the policy is essentially converged by 50k on PushT. (This is a *different*
conclusion from the pre-fix E2 result, where success was flat from 25k onward at a much lower 22–24% — that was
a plateau caused by a broken training distribution, this is a plateau at baseline-matching performance.)

Two caveats on the 175k number: (a) it was selected as the max of 8 noisy n=50 evals, which is an upward-biased
selection, and although it was then re-measured independently at n=200, lerobot's default eval seed (1000)
means the n=200 episode set overlaps the n=50 set it was selected on; (b) the baseline's 175k pick was made
the same way, so the methodologies match — this is a like-for-like comparison of two best-of-periodic-eval
picks, not of two final checkpoints. For reference, final-checkpoint vs. final-checkpoint is 69.0% (ours, 200k)
vs. 62.0% (our local 50-ep eval of the shipped baseline).

Interestingly, longer training makes the policy *more decisive rather than more accurate*. Per-episode
`max_reward` distributions (n=200): the 50k checkpoint has the **highest** `avg_max_reward` (0.971) and the
fewest outright failures (4 episodes below 0.5), but 51 episodes stuck in the [0.95, 1.0) near-miss band; the
175k checkpoint has more clean successes (147 vs. 139 at exactly 1.0) and fewer near-misses (29), but more
catastrophic failures (14 below 0.5) and a *lower* `avg_max_reward` (0.928). Longer training trades a few
episodes it used to nearly-solve for more episodes it fully solves — which is what `pc_success` rewards and
`avg_max_reward` does not.

### A note on comparing at 50k

lerobot's `CosineDecayWithWarmupSchedulerConfig.build()` auto-scales `num_decay_steps` down to the actual run
length when `steps < num_decay_steps`, so a 50k run is a *complete, fully-annealed* run, not a truncated
prefix of a 200k one. That is why the E4b 50k run (69.0%) and the 200k run's own 50k checkpoint (66.0% at
n=50, mid-schedule at lr 8.0e-5) are different things and are reported separately above.

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
collapse. A 200-episode eval of that 50k checkpoint gave **69.0%**, CI [62.6, 75.4]
(wandb [113nyoss](https://wandb.ai/ameya555-ieee/lerobot/runs/113nyoss)) — already at baseline level, which
is what motivated the equal-budget 200k run reported at the top.

**`drop_n_last_frames = horizon - n_action_steps - n_obs_steps + 1` is wrong.** That formula comes from the
original diffusion-policy config (16/8/2 → 7) and does not generalize to `horizon=64`. lerobot's own
`DiffusionConfig` also still ships 7 alongside `horizon=64`/`n_action_steps=32`; the comment claiming the
formula is stale upstream too. The value is now pinned at 7 with this reasoning recorded in
`configuration_flow_matching.py`.

## Confirmed configuration

`n_obs_steps=2`, `horizon=64`, `n_action_steps=16`, `drop_n_last_frames=7`, `crop_shape=(84,84)` with
`crop_is_random=True`, `use_group_norm=True`, `pretrained_backbone_weights=None`, `down_dims=(512,1024,2048)`,
`num_inference_steps=10`, `time_embed_scale=1000.0`, `time_sampling_alpha/beta=1.5/1.0`, batch 64, AdamW
preset with cosine decay + 500 warmup steps, `ode_solver="euler"`. Train with:

```bash
# 50k reaches 69.0% in ~1h05m; 200k reaches 73.5% at its best checkpoint in ~4h20m (not a
# statistically resolvable difference at n=200 -- pick based on how much compute you want to spend).
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

## ODE solver: euler dominates heun and rk4 at matched compute — keep the default

`ode_solvers.py` ships three fixed-step solvers (`euler`, `heun`, `rk4`) selectable via
`FlowMatchingConfig.ode_solver`. Higher-order solvers call the UNet more per step (heun 2x, rk4 4x), so the
fair axis is **NFE = number of denoise_fn calls**, not `num_inference_steps`. All evals below are on the
175k checkpoint, n=200 episodes (n=100 for the two convergence-check rows). euler@NFE10 is the 73.5% headline.

| solver | `num_inference_steps` | NFE | `pc_success` | 95% CI | `avg_max_reward` |
|---|---:|---:|---:|---|---:|
| euler | 10 | 10 | **73.5%** | [67.4, 79.6] | 0.928 |
| euler | 5 | 5 | 61.5% | [54.8, 68.2] | 0.906 |
| euler | 4 | 4 | 64.0% | [57.3, 70.7] | 0.914 |
| heun | 5 | 10 | 7.0% | [3.5, 10.5] | 0.813 |
| heun | 3 | 6 | 2.0% | [0.1, 3.9] | 0.677 |
| heun | 2 | 4 | 0.0% | [0.0, 0.0] | 0.497 |
| rk4 | 3 | 12 | 29.0% | [22.7, 35.3] | 0.896 |
| rk4 | 2 | 8 | 9.0% | [5.0, 13.0] | 0.801 |
| rk4 | 1 | 4 | 1.5% | [0.0, 3.2] | 0.607 |

**At every matched NFE, higher-order solvers are drastically worse** (heun@NFE10 7.0% vs. euler@NFE10 73.5%,
z = -13.6, p < 1e-10; likewise rk4). No solver dominates euler anywhere, least of all at low NFE where a
higher-order win would matter for latency. **The default stays `euler`.**

**This is not a solver bug — it was ruled out two ways.** (1) On a synthetic smooth velocity field, both
heun and rk4 are strictly *more* accurate than euler at matched step count (rk4@5 error 0.011 vs. euler@5
error 0.64), confirming the implementations are correct classical integrators. (2) On the real policy, heun
*converges to euler's result as steps grow* — `num_inference_steps` 5 → 20 → 40 gives 7% → 53% → 71%,
approaching euler@40's 76% — which is exactly what a consistent solver must do, so the low-NFE collapse is a
property of the *learned field*, not the code.

The mechanism: heun and rk4 spend their extra UNet calls evaluating the velocity at *predicted, off-path*
points (`x + dt·v`, `x + dt/2·k`, ...). During training the network only ever saw on-path samples
`x_t = t·noise + (1-t)·action`, so its velocity is only reliable near that manifold; a large-`dt` predictor
step lands off it, the corrector reads an unreliable velocity there, and averaging makes the step *worse*
than the plain euler march that never leaves the manifold. Only once `dt` is small enough that the predicted
point stays near-path (num_steps ≳ 40, i.e. NFE ≳ 80) does heun recover — at 8x euler's cost for no benefit.
This is the same reason diffusion samplers use purpose-built solvers (DPM-Solver et al.) rather than generic
RK4. Repro: `scripts/eval_pretrained.sh <ckpt> pusht 200 <out> -- --policy.ode_solver=heun --policy.num_inference_steps=5`.

These 11 solver evals are logged to wandb as `m4b_175k_{solver}_nfe{N}` and `conv_*` (also stored under
`outputs/eval/m4b_175k_*` / `outputs/eval/conv_*`); `scripts/log_solver_sweep_to_wandb.sh` re-pushes them.

### Follow-up (E5): off-path training jitter does not rescue higher-order solvers — and hurts euler

Hypothesis: if the field is only reliable *on* the interpolant path, train it to be reliable in a
*neighborhood* of the path. The `path_noise_std` config field adds isotropic Gaussian jitter of that std to
the training-time point `x_t` before the UNet, keeping the target velocity `noise - actions` unchanged.
`outputs/train/e5_path_noise_005` is a 50k run with `path_noise_std=0.05`, otherwise the confirmed recipe
(wandb [ab2hnfq7](https://wandb.ai/ameya555-ieee/lerobot/runs/ab2hnfq7)). The jitter scale is well-chosen,
not too small: instrumenting the sampler, heun's predictor lands **0.028–0.047** (per-element RMS) off the
true path at NFE10-equivalent steps, right where 0.05 covers.

All e5 evals n=200, on the 50k checkpoint. Matched-NFE comparison against the euler-trained 175k checkpoint
from the table above (different checkpoint — see the euler-vs-euler control below):

| solver | NFE | e5 (path_noise 0.05) | euler-trained 175k | Δ | two-prop p |
|---|---:|---:|---:|---:|---:|
| euler | 10 | 52.0% [45.1, 58.9] | 73.5% | −21.5pp | <1e-4 |
| heun | 10 | 4.0% [1.3, 6.7] | 7.0% | −3.0pp | 0.19 |
| heun | 6 | 2.0% | 2.0% | 0.0pp | 1.0 |
| rk4 | 12 | 23.0% [17.2, 28.8] | 29.0% | −6.0pp | 0.17 |
| rk4 | 8 | 14.0% [9.2, 18.8] | 9.0% | +5.0pp | 0.12 |
| rk4 | 4 | 1.0% | 1.5% | −0.5pp | 0.65 |

**The hypothesis did not pan out.** Every heun/rk4 point is statistically indistinguishable from the
euler-trained checkpoint (all p ≥ 0.12) — the jitter left higher-order solvers exactly on the floor. euler
still dominates on the very same weights (52% vs. heun 4%, rk4@12 23%).

**Worse, the jitter significantly degraded euler itself.** The clean control is euler-vs-euler at matched
50k steps (the 175k comparison above is confounded by checkpoint): E4b 50k (`path_noise_std=0`) scored 69.0%
[62.6, 75.4] vs. e5 50k (`path_noise_std=0.05`) at **52.0%** [45.1, 58.9] — a 17pp drop, z = −3.48,
p = 0.0005. (E5's own in-training n=50 eval read 66%, but n=50 is too noisy; the n=200 number is 52%.)

Why it fails, mechanistically: the correct off-path velocity is the *marginal* `E[noise − action | x_t]`,
which differs from any single sample's `noise − action` once you leave that sample's own path. Jittering
`x_t` while keeping this sample's target feeds the network a *biased* velocity at the jittered point, so it
doesn't learn the true off-path field heun/rk4 need — it just blurs the on-path field toward locally
constant, which slightly dulls euler's on-path precision (the 17pp cost) without supplying correct off-path
velocities. Bottom line: **`path_noise_std` stays at its default 0.0; this route to higher-order solvers is
a dead end as formulated.** A correct version would need the jittered point's own marginal target (e.g.
mini-batch OT / a consistency-style objective), not the single-sample target — a much larger change, not
pursued.

e5 evals logged as `e5_{solver}_nfe{N}`: euler [xubwb1y2](https://wandb.ai/ameya555-ieee/lerobot/runs/xubwb1y2),
heun@10 [ffkpgram](https://wandb.ai/ameya555-ieee/lerobot/runs/ffkpgram),
rk4@12 [0xoyu6ni](https://wandb.ai/ameya555-ieee/lerobot/runs/0xoyu6ni) (others under the same prefix).

## Open items

None of these block M4; all are cheap (eval-time) except where noted.

- **Re-test `num_inference_steps` and the time-sampling distribution** on the fixed config. The 10-vs-100
  result below was measured on the old, broken-augmentation checkpoint and should not be assumed to carry
  over. A `{5, 8, 10, 15, 20, 30}` sweep is eval-time only; uniform-vs-Beta time sampling needs a 50k retrain
  (~1h05m, now cheap given that 50k is enough).
- **Remaining headroom is split between near-misses and hard failures**, and the balance shifts with training
  length: the 50k checkpoint leaves 51/200 episodes in `max_reward` ∈ [0.95, 1.0) with only 4 hard failures;
  the 175k checkpoint has 29 near-misses but 14 hard failures. Whether those 14 are a distinct failure mode
  (e.g. specific initial block poses) hasn't been checked — the rollout videos are saved and would answer it.
- **`time_embed_scale` has still never been swept** — the default worked on the first try at every
  milestone, which is not the same as tuned.

## Conclusion

At equal training budget (200k steps) and matched checkpoint-selection methodology (best of periodic
50-episode evals, which is how the baseline picked its own 175k checkpoint), the Flow Matching policy reaches
**73.5% PushT success over 200 episodes (CI [67.4, 79.6])** against the baseline's published **65.4% over
500 episodes (CI [61.2, 69.6])** — a marginal edge (p = 0.038), achieved with **1/10 the sampling steps per
action chunk** (10 Euler steps vs. ~100 DDPM steps) and ~1.7x faster rollouts. Honest summary: **parity, with
a possible modest edge, at a large inference-cost advantage.**

A secondary finding worth as much as the headline: **training past 50k steps is not resolvable.** A fully
annealed 50k run scores 69.0% (n=200) versus 73.5% for the best checkpoint of a 200k run (p = 0.32) — 4x the
compute for no statistically detectable gain.

The architecture is unchanged from `DiffusionPolicy` apart from the generative process, and no `diffusers`
import exists anywhere in the policy. Every performance gap found between the two policies during M4 traced
to configuration — execution horizon (`n_action_steps`), image augmentation (`crop_shape`/`use_group_norm`),
and the `drop_n_last_frames` sampler bug — not to flow matching itself. The M4 gate is met.
