#!/bin/bash
# S10 session 2 (track F): multiple shooting on jaccpot's fused production lane -- the timings, on an exclusive
# HoreKa accelerated node (4x A100-40GB). bench/s10_fused_shooting.py --task timing, one row (one program kind, one
# configuration) per process. Resumable: a row whose JSON exists is skipped, so a resubmission continues where the
# last one stopped (a failed row leaves no JSON and runs again).
#
#   sbatch bench/slurm/s10_shooting_timing.sbatch                  # the whole matrix
#   ROWS='^seg_' sbatch bench/slurm/s10_shooting_timing.sbatch     # the rows whose tag matches a regex
#   python bench/slurm/s10_shooting_summary.py $OUT/rows           # the table
#
# THE SYSTEM (bench/s10_fused_shooting.py's docstring): a self-gravitating disc (mass 1e-2) in nornax's Miyamoto-Nagai
# + NFW potential, theta = the halo's (log M, log r_s), FastMultipoleMethod(preset="galaxy") (dehnen_error 1e-4, leaf 64,
# p6, float32), softening 0.015 (N / 1e5)^(-1/3), dt 0.01, S segments of L steps, FusedSegmentEngine, checkpoint "step".
#
# What the rows answer:
#   pre_<N>_fc1 / pre_<N>_fc0   a warm pre-pass (FusedSegmentEngine.propagate: S runner.gradient_plan calls at new
#                               nodes each time) with and without gradient_plan(freeze_cache=...): what a re-plan costs
#                               the optimizer loop; the cold one (compiles) recorded beside it.
#   seg_<N>                     each segment's value-and-gradient program (sum(w . x_L) + sum(w . v_L) in theta, x_k,
#                               v_k): ms per step, against S9's forward / gradient rows of jaccpot's
#                               grad_rollout_timing.sbatch at the same N.
#   al_<N>_one_<opt>            ONE augmented-Lagrangian inner iteration as the driver runs it (prepare, step, check,
#                               accept) with the whole objective in one program on one card (S = 2 segments in it);
#                               L-BFGS with optax's zoom line search inside the step.
#   al_<N>_c<K>_<opt>           the same iteration in mimirax's per-segment mode (track P, AugmentedLagrangian(devices=
#                               K cards)): one value-and-gradient program per segment, segment k on card k mod K, the
#                               host sums; S = K (K = 2, 4), and S = 2 on one card (c1). L-BFGS without a line search in
#                               the update, Armijo backtracking on the host (each trial one parallel forward pass; the
#                               trials per iteration are in the row).
#
# MEMORY (40 GB cards; measured locally, F report): one segment's gradient program is W + L r with r 16 B/p per step and
# W (this engine, GradientPlan.rollout: the start force outside the adjoint) 5088 B/p at 1e5, 2382 at 1e6 (cml 8:
# 1-3 particles per leaf on this disc), ~616 at 2.5e7 (L 10: 776 B/p of temporaries, peak 966 B/p = 22.5 GiB with the
# preallocated BFC; without preallocation the 18 GiB block did not fit after the pre-pass). The one-card one-program
# step holds every segment's residuals (W + S L r + ~400 B/p of AL terms): not at 2.5e7. Per segment, card 0 also
# holds the parameters, multipliers, data and optimizer state (Adam ~200 B/p; L-BFGS +528 B/p): L 20 and Adam only
# at 2.5e7. Every row runs with the preallocated BFC (BFC below).
#
#SBATCH --job-name=mimirax-s10-shooting-timing
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# --- adapt: a python with jax 0.11.2 (CUDA 12), jaccpot (s10/fused-gpu or later), yggdrax, nornax
#     (s10/shooting-potentials or later) and this mimirax checkout importable
# source ~/envs/jaccpot-jax0112/bin/activate
MIMIRAX=${MIMIRAX:-$PWD}
OUT=${OUT:-${WORKSPACE:-$HOME}/s10-shooting-timing-$(date +%Y%m%d)}
ROWS=${ROWS:-.}
PY=${PY:-python}
REPS=${REPS:-5}

ALL_CARDS=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
echo "node $(hostname) cards ${ALL_CARDS}"
nvidia-smi || true
mkdir -p "$OUT/rows" "$OUT/logs"
cd "$MIMIRAX"
git rev-parse HEAD > "$OUT/mimirax_commit.txt" 2>/dev/null || true
export JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1
# bitwise reruns: the fusion autotuner picks emitters by timing (jaccpot docs/production.md, "Launch environment")
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_experimental_enable_fusion_autotuner=false"

S=bench/s10_fused_shooting.py
BFC="XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.95"
cards() { echo "$ALL_CARDS" | cut -d, -f1-"$1"; }

# tag | cards | script arguments (the row's JSON is $OUT/rows/<tag>.json)
ROW_LIST=()
for n in 1e5 1e6 2p5e7; do
  case $n in
    1e5) N=100000; L=50; OPTS="adam lbfgs" ;;
    1e6) N=1000000; L=50; OPTS="adam lbfgs" ;;
    2p5e7) N=25000000; L=20; OPTS="adam" ;;
  esac
  base="--task timing --n $N --steps $L --reps $REPS"
  ROW_LIST+=(
    "pre_${n}_fc1|1|$base --segments 2 --only prepass"
    "pre_${n}_fc0|1|$base --segments 2 --only prepass --no-freeze-cache"
    "seg_${n}|1|$base --segments 2 --only seggrad"
  )
  for opt in $OPTS; do
    # one program (the S segments sequential, optax's own line search for L-BFGS): not at 2.5e7 (memory)
    if [ "$n" != 2p5e7 ]; then
      ROW_LIST+=("al_${n}_one_${opt}|1|$base --segments 2 --only alstep --optimizer $opt")
    fi
    # per segment (track P's mode; L-BFGS with the host's Armijo backtracking): S = K segments on K cards,
    # and S = 2 on one card (the per-segment programs one after the other: W + L r each, not W + S L r)
    ROW_LIST+=("al_${n}_c1_${opt}|1|$base --segments 2 --devices 1 --only alstep --optimizer $opt")
    for k in 2 4; do
      ROW_LIST+=("al_${n}_c${k}_${opt}|$k|$base --segments $k --devices $k --only alstep --optimizer $opt")
    done
  done
done

for row in "${ROW_LIST[@]}"; do
  IFS='|' read -r tag ncards cmd <<< "$row"
  [[ $tag =~ $ROWS ]] || continue
  json=$OUT/rows/$tag.json
  if [ -s "$json" ]; then echo "skip $tag (done)"; continue; fi
  echo "=== $tag  [cards $(cards "$ncards")]  $(date -Is)"
  start=$(date +%s)
  # shellcheck disable=SC2086
  if env $BFC CUDA_VISIBLE_DEVICES="$(cards "$ncards")" $PY $S $cmd --out "$json" > "$OUT/logs/$tag.log" 2>&1; then
    status=ok
  else
    status="FAILED rc=$?"
  fi
  echo "    $status in $(( $(date +%s) - start )) s (log $OUT/logs/$tag.log)"
done

$PY bench/slurm/s10_shooting_summary.py "$OUT/rows" | tee "$OUT/summary.txt" || true
