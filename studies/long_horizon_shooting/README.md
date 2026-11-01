# When is multiple shooting worth its cost? A chaotic N-body horizon

S10 session 2, track L (2026-11-01). Session 1 found that on SHORT horizons multiple
shooting costs far more than single shooting for the same answer (a Plummer field, N 32,
80 steps: 15 inner L-BFGS steps against 1834 with 4 segments). The case for multiple
shooting is a LONG, CHAOTIC horizon, where the single-shot objective turns rough. This
study measures whether and where that cross-over happens, on the direct sum in float64
(so chaos is not confounded with round-off), N = 1024.

**Answer in one paragraph.** On this chaotic N = 1024 system (lambda = 0.35 / t_cross, a Lyapunov time
of 83 steps) single shooting is the right tool up to ~8 Lyapunov times: one smooth basin
across a 10 % window, 10/10 recoveries in ~450 segment passes, 8-10x cheaper than multiple
shooting. Between 8 and 10 Lyapunov times its objective turns rough -- the gradient grows
like `exp(lambda T)`, dozens of spurious minima appear, the true basin shrinks to ~1e-4 --
and its success falls to 7/10, 3/10 and 0/10 at 9.7, 11.6 and 15.5 Lyapunov times. There,
multiple shooting with nodes initialised from the observed states recovers the halo (3/3 at
each horizon, theta within 4e-4) at 5,000-13,000 passes per fit; with nodes propagated from
the initial guess it fails like single shooting (0/6). Multiple shooting is worth its cost
beyond ~10 Lyapunov times, provided its nodes can start near the true trajectory.

## The system

A self-gravitating stellar system at the centre of session 1's analytic galaxy
(`common.py`): a Plummer sphere of N = 1024 equal-mass stars (mass 0.2, scale a = 0.3,
G = 1) inside nornax's Miyamoto-Nagai disc field (M 1, a 1, b 0.1; fixed) plus an NFW
halo (M 5, r_s 3) whose `theta = (log M, log r_s)` is the unknown. Inside the stars the
galaxy's pull and their own are of the same order (the galaxy's virial term is 1.3x the
stars' own), so the halo acts on every orbit at order one. The Plummer velocities are
scaled by 1.55 so that 2K = W in the total potential. Half-mass radius 0.41, rms speed
0.68, crossing time t_cr = r_h / v_rms = 0.60. Softening 0.05 a = 0.015 (jaccpot's default
`ferrers3` kernel, Plummer-equivalent length), dt = 0.02 (0.033 t_cr). Segment engine:
session 1's `DirectSumSegmentEngine` (jaccpot `differentiable_rollout` over the softened
direct sum, `rows_field(galaxy)`, `checkpoint="step"`; `checkpoint=None` for tangents).

**Why this system, by measurement** (`lyapunov.py`, candidates at N = 1024, 40 t_cr):

| candidate | lambda t_cr | steps per Lyapunov time | log dX/dtheta - lambda t at 4 / 8 / 12 t_lambda |
|---|---|---|---|
| embedded (chosen) | 0.37 | 81 | +1.13 / +0.52 / +0.46 |
| cluster on a circular orbit at R = 2 (mass 0.01, a 0.05; t_cr its own) | 0.88 | 57 | +0.06 / -2.51 / -3.24 |
| self-gravitating disc (mass 0.1, R 0.5-2, softening 0.02; t_cr = 1/Omega(R=1)) | 1.06 | 67 | +0.11 / -1.13 / -1.15 |

All three are chaotic, and in all three the halo's tangent `dX/dtheta` eventually grows
at the Lyapunov rate. The last column is how strongly the halo is coupled into that
growth: `log |dX/dtheta|` minus `lambda t` (dimensionless norm, `x / r_h`, `v / v_rms`).
In the embedded system it stays at +0.5 from 4 Lyapunov times on -- the halo feeds the
unstable directions at order one from the start. The orbiting cluster's halo tangent first
grows like `t^2` (the regular drift of its centre of mass along its orbit) and joins the
exponential only after ~8 Lyapunov times, 3.8 e-folds (45x) lower; the disc's sits 1.6
e-folds lower. The cost per Lyapunov time in steps is similar for all three (57-83). The
embedded system is the clean test: the halo's information reaches the data through the
chaotic dynamics, not through a regular channel that would give single shooting a smooth
funnel to follow.

## 1. The Lyapunov exponent

Largest exponent from a renormalised random tangent (`jax.jvp` through the
`checkpoint=None` rollout, renormalised every 10 steps; the mean log growth over the
second half of a 40 t_cr run), three seeds per point:

| N | softening 0.025 a | 0.05 a | 0.1 a | 0.2 a |
|---|---|---|---|---|
| 256 | 1.22 +- 0.12 (49) | 0.68 +- 0.01 (43) | 0.31 +- 0.02 (96) | 0.17 +- 0.01 (180) |
| 512 | 1.03 +- 0.11 (57) | 0.51 +- 0.03 (58) | 0.24 +- 0.01 (123) | 0.14 +- 0.01 (216) |
| 1024 | 0.70 +- 0.01 (83) | **0.35 +- 0.02 (83)** | 0.19 +- 0.01 (155) | 0.14 +- 0.05 (231) |
| 2048 | 0.49 +- 0.00 (116) | 0.26 +- 0.02 (111) | 0.14 +- 0.01 (206) | 0.09 +- 0.00 (339) |

`lambda t_cr`, mean +- standard deviation over three seeds; in brackets the steps per
Lyapunov time (dt 0.01 at 0.025 a, 0.02 otherwise). Each IC is rebuilt per seed and
softening (its own virial scaling and crossing time, t_cr 0.55-0.63). At 0.2 a the
second-half estimate has not settled in 40 t_cr (the halo's tangents grow at only 0.35-0.8
lambda there).

The chosen configuration: **lambda = 0.605 per unit time = 0.35 / t_cr, a Lyapunov
time of 1.65 (83 steps)**; one 160-step segment is 1.94 Lyapunov times. lambda
does not depend on the timestep (dt 0.005 / 0.01 / 0.02 / 0.04: lambda 0.60 / 0.56 / 0.54 / 0.58, 20 t_cr runs of seed 0
-- the shorter run's own scatter;
energy drift 2.5e-7 / 5.6e-7 / 1.5e-5 / 2.8e-4). The halo's tangents grow at 0.95-1.06
lambda over the second half for every softening <= 0.1 a.

**Softening sets lambda.** Over the grid (softening 0.025-0.1 a, N 256-2048) lambda t_cr
= 0.385 N^-0.43 (softening / a)^-0.98 (36 runs, rms 7 % in log). Halving the softening doubles the exponent; at fixed softening a larger N is
less chaotic (the encounters at the softening scale that drive the divergence are weaker
for lighter stars). The cost of a horizon in steps follows: halving the softening also
needs half the timestep, so the steps per Lyapunov time hardly change (83 at 0.025 a and
at 0.05 a, N 1024); a softer system costs more steps per Lyapunov time (155 at 0.1 a).

## 2. Roughness of the objective against the horizon

`roughness.py`. The data: the full phase-space state every 160 steps (the end of every
segment), Gaussian noise sigma = 1e-3 on every coordinate; a horizon of S segments sees
the first S snapshots of one time series (S = 1 .. 16, T = 1.9 .. 31 Lyapunov times).
Two objectives, both scanned along log M, log r_s and a fixed random direction through
the truth:

* **single shooting**: `f_S(theta)`, the negative log-likelihood along one trajectory;
* **multiple shooting**: the augmented Lagrangian at zero multipliers, `mu = 1 / sigma^2`
  (a defect weighs what a datum does), **nodes held at the truth's propagated states**.
  Why those nodes: at nodes propagated at `theta` itself (`init_nodes(theta)`) every
  defect vanishes and the Lagrangian IS `f_S` -- the scan would repeat single shooting;
  optimised nodes (the profile) need an inner optimisation per scan point. Held at the
  truth, the scan is the slice the multiple-shooting optimizer works in once its nodes
  are near the trajectory: every segment is 1.9 Lyapunov times long whatever S is.

| S | T / t_lambda | single: minima in +-0.1 (random / log M / log r_s) | single: basin half-width (random) | grad single | grad multiple | single: Fisher sigma |
|---|---|---|---|---|---|---|
| 1 | 1.9 | 1 / 1 / 1 | >= 0.1 | 4.1e6 | 4.1e6 | 5.2e-4 |
| 2 | 3.9 | 1 / 1 / 1 | >= 0.1 | 3.5e7 | 1.2e7 | 1.5e-4 |
| 3 | 5.8 | 1 / 1 / 1 | >= 0.1 | 4.8e8 | 2.0e7 | 3.0e-5 |
| 4 | 7.7 | 1 / 1 / 1 | >= 0.1 | 3.5e9 | 2.7e7 | 5.5e-6 |
| 5 | 9.7 | 6 / 10 / 5 | 2.1e-3 | 2.4e10 | 3.4e7 | 8.1e-7 |
| 6 | 11.6 | 46 / 60 / 37 | 1.2e-3 | 7.5e10 | 4.1e7 | 1.3e-7 |
| 7 | 13.6 | 69 / 82 / 76 | 3.2e-4 | 4.8e11 | 4.8e7 | 2.4e-8 |
| 8 | 15.5 | 93 / 95 / 96 | 3.2e-4 | 3.1e12 | 5.5e7 | 3.1e-9 |
| 10 | 19.4 | 106 / 119 / 106 | 1.8e-4 | 3.1e13 | 7.0e7 | 1.0e-10 |
| 12 | 23.2 | 112 / 116 / 114 | 1.3e-4 | 1.6e14 | 8.5e7 | 3.4e-12 |
| 16 | 31.0 | 123 / 115 / 124 | 2.5e-5 | 1.9e18 | 1.1e8 | 1.9e-15 |

Multiple shooting (nodes at the truth): one minimum and a basin covering the whole window
at every S along all three directions. "grad": `|grad_theta|` at `|delta theta| = 1e-3`
along the random direction. "Fisher sigma": the larger of the two forecast standard
errors `sqrt(diag F^-1)`, `F = J^T J / sigma^2` of single shooting's data at the truth.

* **Gradient** (exact, `jax.jacfwd` through `checkpoint=None`, at `|delta theta| = 1e-3`
  along the random direction): single shooting grows from 4.1e6 to 1.9e18 over 31
  Lyapunov times -- 1.7 e-folds per segment, the Lyapunov rate -- while multiple shooting
  grows LINEARLY in S (4.1e6 to 1.1e8: each segment adds its own bounded share).
* **Spurious minima** in `|delta theta| <= 0.1` (401 points): single shooting has exactly
  one minimum up to S = 4 (7.7 Lyapunov times), then 6, 46, 69, 93 at S = 5, 6, 7, 8 and
  100-123 beyond (the grid's resolution, 5e-4, caps the count; the fine window
  `|delta| <= 0.004`, 2.5e-5 apart, has 98 at S = 16). Multiple shooting: one, at every
  horizon, along every direction.
* **Basin of the true minimum** (walking out from the minimum until the objective first
  decreases): single shooting's covers the whole window (>= 0.1) up to S = 4, then shrinks
  to 2.1e-3 (S 5), 1.2e-3 (S 6), 3.2e-4 (S 7-8) and ~1e-4 by S 11 (the merged grid's
  resolution there is ~2.5e-5-5e-5). Multiple shooting's covers the window at every S.
* **The information** in the data grows exponentially too: the Fisher forecast of single
  shooting's error falls from 5e-4 (S 1) to 1e-10 (S 10) and 2e-15 (S 16). A long chaotic
  horizon pins the halo far beyond any tolerance anyone needs -- but its minimum is
  ~1e-4 wide, and nothing that starts further out finds it by descent.

The cross-over in the landscape is sharp: **between 8 and 10 Lyapunov times** (S 4 -> 5)
the single-shot objective goes from one smooth basin across the whole window to a field of
minima.

## 3. Recovery from perturbed starts

`recovery.py`. Starts `theta_0 = theta* + 0.1 u_k` (`u_k` a random unit vector, seed
`1000 + k`: 10 % in the halo's mass and scale together), the data of section 2 (the same
noisy snapshots for every method at a horizon). One driver for every method:
`AugmentedLagrangian(optax.lbfgs(), inner_steps=200, max_outer=20, tol=1e-8,
mu0=1/sigma^2)` -- session 1's rules otherwise (penalty x10 when the violation does not
shrink 4x, cap 1e12). Three methods:

* **single**: `{"theta"}` through `SeriesModel` (one trajectory, all snapshots); no
  defects, so the driver is plain L-BFGS restarted every outer iteration;
* **multiple_data**: `{"theta", "nodes"}`, nodes initialised from the DATA (node k = the
  observed, noisy state at its time) -- the classical multiple-shooting start (Bock 1981),
  available because the whole phase-space state is observed;
* **multiple**: the same, nodes from a forward run at the START theta (`init_nodes`,
  defects zero), as the brief specified.

**Success**: `max |theta - theta*| <= 1e-3` at the end (100x closer than the start).
**Cost**: segment passes (one segment's value and gradient, line-search trial points
included; counted by a host callback), NOT wall time. A single-shooting evaluation is S
passes, a multiple-shooting one S passes too.

**Measurement budget (a deviation, said loudly).** The cards were shared by three tracks
and a single-card job waited up to 2 h for a slot; multiple-shooting fits are 10-100x
dearer than single-shooting ones. So: single shooting has K = 10 starts at every horizon;
multiple shooting has K = 3 (multiple_data at S 5 uses the three starts single shooting
FAILED on, k = 2, 4, 9; elsewhere k = 0-2), and every multiple-shooting fit was stopped
by a WALL-TIME CAP (`--cap-min`, status "time cap": 14 min; 40 min for S 8 and S 6
k = 1; S 6 k = 0 ran 50 min under the batch's process limit, before progress histories
were recorded) long before
the driver's own stopping rule (violation <= 1e-8 and a stagnant objective), which no
multiple-shooting fit reached within the caps. Its "success" is therefore theta at the
cap, and its cost is the passes after which theta STAYED within the tolerance (from the
fit's progress history, every 25 inner steps). That reading favours multiple shooting
(an oracle stop); single shooting's costs are its driver's own full runs.

| S | T / t_lambda | single (K = 10): successes, passes per success | multiple_data (K = 3): successes, passes to tolerance | multiple, forward nodes (K = 3) |
|---|---|---|---|---|
| 2 | 3.9 | 10/10, 116-126 (all to 3.1e-4, the global minimum) | -- | pilot, 1 start: converged, 7466 passes |
| 4 | 7.7 | 10/10, 396-480 (2.6e-6) | 2/3 at the 14-min cap, 3792 / 4600 (1.4e-4, 5.5e-4; k1 at 1.4e-3) | -- |
| 5 | 9.7 | 7/10, 670-1015 (one 11250) | 3/3 on single's 3 FAILED starts, 5020-6880 (2.2e-4-3.5e-4) | -- |
| 6 | 11.6 | 3/10, 1338 / 4560 / 8550 | 3/3, 6048 / 7992 / (k0 before histories: within 2.1e-4 by 7536) | 0/3, 0.074-0.078 at ~4900 |
| 8 | 15.5 | 0/10 (ends 0.044-0.22, one 5.5) | 3/3 at the 40-min cap, 6640 / 8088 / 13408 (9.2e-5-2.0e-4 at ~31000) | 0/3, 0.095-0.17 at ~4800; pilot 0.046 after the full budget (65328) |

Errors in brackets are `max |theta - theta*|` at the end. "Passes per success" for single
shooting is its driver's full run. Medians of the inner steps per fit: single 20 / 24 / 34
/ 41 / 54 at S 2-8; multiple_data 600 / 625 / 2450 / 1825 (the caps, not convergence).

* **Single shooting collapses where the landscape does.** 10/10 at 3.9 and 7.7 Lyapunov
  times, 7/10 at 9.7, 3/10 at 11.6, 0/10 at 15.5. Its failures stop in wrong minima
  0.04-0.1 from the truth (the driver "converges": the objective is stagnant there, 1e8-3e9
  above its value at the truth); two S 8 starts wandered (log M off by 5.5 at k = 1) while
  the line search failed step after step. Started AT the truth, S 8 single shooting
  leaves the basin on its first step and ends at log M = 153 (11624 passes): with a
  gradient of ~1e12 and a basin ~1e-4 wide, no line search finds a decrease.
* **Multiple shooting from the data holds there.** 3/3 at 9.7 Lyapunov times on the three starts single
  shooting failed on, 3/3 at 11.6 and 3/3 at 15.5 (theta within 0.9-2.0e-4 of the truth at
  the cap), where single shooting has 3/10 and 0/10. It needs 5000-13400 passes to settle
  within 1e-3 -- 5-10x the passes of a successful single-shooting fit at S 5 -- and its
  defects were still above the 1e-8 tolerance at every cap: the driver had not finished.
  Its landscape explains why it holds: section 2's slice has one minimum and a basin as
  wide as the window at every horizon, and each segment's gradient stays ~1e7.
* **Multiple shooting from a forward run does not.** Nodes propagated at the start theta
  are as decorrelated from the data as single shooting's trajectory: 0/3 at S 6 (0.074-0.078
  after ~4900 passes) and 0/3 at S 8 (0.095-0.17), and the 8-segment pilot (mu0 1e6,
  the full 20 x 200 budget, 65328 passes, 38 min alone on a card) ended at 0.046 with the
  defects still at 0.38. On a short horizon it works (pilot, S 2: converged, theta within
  1e-7 of single shooting's, 7466 passes against single shooting's 126).
* **The price.** Where both succeed (S 4) single shooting needs 396-480 passes (24 inner
  steps) and multiple shooting from the data 3800-4600 to stay within 1e-3 -- and its
  estimate is the cruder one (1.4e-4-5.5e-4 against single shooting's 2.6e-6, the
  global minimum, at the cap). At S 5-6 multiple shooting from the data needs 5000-7000
  passes for an answer single shooting reaches in 670-1340 when it reaches it at all.

## 4. When is multiple shooting worth its cost?

**Multiple shooting is worth its cost exactly where single shooting's objective has turned
rough, and only if its nodes start near the trajectory.** Measured on this system:

1. **Below ~8 Lyapunov times use single shooting.** Its objective has one minimum in
   `|delta theta| <= 0.1`, 10/10 starts recover the halo to the global minimum (2.6e-6 at
   7.7 Lyapunov times) in ~450 passes, and multiple shooting costs 8-10x more for a cruder
   answer (session 1's short-horizon finding, reproduced at N = 1024).
2. **The cross-over is at 8-10 Lyapunov times** (here 650-800 steps; one Lyapunov time is
   83 steps at softening 0.05 a). The single-shot gradient grows as `exp(lambda T)`
   (1.7 e-folds per 1.94-Lyapunov-time segment), spurious minima appear between 7.7 and 9.7
   Lyapunov times (1 -> 6 -> 46 -> 93 at S 4-8), the true basin shrinks from >= 0.1 to
   2e-3 and then ~1e-4, and single shooting's success falls 10/10 -> 7/10 -> 3/10 -> 0/10.
   From 16 Lyapunov times even a start AT the truth leaves the basin.
3. **Beyond it, multiple shooting from data-initialised nodes recovers** (3/3 at 9.7, 11.6
   and 15.5 Lyapunov times), at 5,000-13,000 segment passes per fit -- far more than a
   short-horizon single shot, but the long horizon is out of single shooting's reach at
   any price. The segment length, not the horizon, sets its landscape: keep segments at
   ~2 Lyapunov times.
4. **Multiple shooting with nodes from a forward run at the guess inherits the chaos it is
   meant to avoid**: the propagated nodes are decorrelated from the data, and 6/6 fits
   stall 0.07-0.17 from the truth (the full-budget pilot: 0.046). The node initialisation
   is the method. Here the whole phase-space state is observed, so data-initialised nodes
   are free; with partial observations (projected positions, line-of-sight velocities)
   they need another estimate of the state, and that -- not the segment arithmetic -- is
   the open problem for potential inference on long horizons.
5. **The price is iterations, not landscape.** The augmented Lagrangian with a VJP-only
   L-BFGS inner loop moves 6N(S-1) node variables (43,008 at S 8) together with two halo
   parameters; no multiple-shooting fit met the driver's own stopping rule (defects
   <= 1e-8) inside 40 minutes. A Gauss-Newton / condensing inner solver, the classical
   companion of multiple shooting, needs forward-mode or per-segment Jacobians that
   jaccpot's checkpointed `custom_vjp` rollouts do not offer; with them the cost gap
   should shrink, and track P's parallel segments divide its wall time by the device count.

**Softening is the dial**: lambda scales as `softening^-1` (and `N^-0.43` at fixed
softening), so the cross-over horizon in physical time moves with the force resolution --
a production run at small softening is chaotic sooner, a heavily softened one later.

**Caveats.** One system, one noise realisation, one initial condition; K = 3 for multiple
shooting (time on the shared cards); multiple-shooting results are at a wall-time cap, its
"success" and cost read off theta's history, not off a converged driver; the success
tolerance (1e-3) is far coarser than the long horizons' statistical error (Fisher sigma
3e-9 at 15.5 Lyapunov times), which neither method reaches.

## Reproduce

```bash
export LONG_HORIZON_RUNS=/export/scratch/tbuck/s9-scratch/s10b/l/runs   # result files
JAX_PLATFORMS=cuda,cpu python studies/long_horizon_shooting/lyapunov.py   # ~2 min GPU (CPU: ~1 h)
JAX_PLATFORMS=cuda,cpu python studies/long_horizon_shooting/roughness.py  # ~15 min GPU
JAX_PLATFORMS=cuda,cpu python studies/long_horizon_shooting/recovery.py \
    --method single --segments 2,4,5,6,8 --starts 0-9
JAX_PLATFORMS=cuda,cpu python studies/long_horizon_shooting/recovery.py \
    --method multiple_data --segments 6,8 --starts 0-2 --cap-min 40   # etc.: see section 3
python studies/long_horizon_shooting/summary.py   # JSON + figure, CPU, seconds
```

Every script skips configurations whose result file exists, so any of them can be
stopped and rerun. `cpu` must be in `JAX_PLATFORMS` for `recovery.py` (its pass
counter is a host callback). The test that pins the qualitative claim cheaply is
`tests/integration/test_long_horizon_shooting.py` (CPU, 19 s).
