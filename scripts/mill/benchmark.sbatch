#!/bin/bash
#SBATCH -p gpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o /share/ceph/scratch/%u/dcs2/logs/%x-%j.out
# NOTE: #SBATCH lines do not expand $S, so the -o path spells out the scratch root (S=/share/ceph/scratch/$USER/dcs2).
#
# Mill denoiser benchmark (#7, used by #8). Submit from the repo checkout (the job runs there). Environment:
#   HW            required. config/hw profile: h100, v100 or rtx3060.
#   DATA          required. low_res dataset directory (contains train/ and test/), e.g. $S/v1/data/low_res.
#   BATCH_SIZES   optional, default "32 64 128". Whitespace-separated.
#   STEPS         optional, default 200. Optimizer steps per batch size.
#   GPUS          optional, informational only (logged). scripts/benchmark_denoiser.py is single-process, so it
#                 measures one GPU even when the job was given more.
#   REPO_DIR      optional, default $SLURM_SUBMIT_DIR (else $PWD). Must contain scripts/benchmark_denoiser.py.
#   DRY_RUN=1     print the planned python command and exit.
# Submit with: sbatch --gres=$H100_GRES -J bench-h100 --export=ALL,HW=h100,DATA=...,BATCH_SIZES="32 64 128",STEPS=200 \
#                     scripts/mill/benchmark.sbatch
# The script prints one 'BENCH {json}' line per batch size, so `grep -h '^BENCH ' $S/logs/bench-*.out` collects them.
set -euo pipefail

die() {
    echo "benchmark.sbatch: ERROR: $*" >&2
    exit 2
}

DRY_RUN=${DRY_RUN:-0}
if [[ -f "$HOME/.dcs2_env" ]]; then
    set +u
    # shellcheck disable=SC1091
    source "$HOME/.dcs2_env"
    set -u
elif [[ "$DRY_RUN" != 1 ]]; then
    die "$HOME/.dcs2_env not found. Copy scripts/mill/env.example.sh there and edit it (docs/MILL.md)."
fi

for var in HW DATA; do
    [[ -n "${!var:-}" ]] || die "$var is not set (pass it with sbatch --export=ALL,HW=...,DATA=...)."
done
BATCH_SIZES=${BATCH_SIZES:-"32 64 128"}
STEPS=${STEPS:-200}
GPUS=${GPUS:-1}
[[ "$GPUS" =~ ^[0-9]+$ ]] || die "GPUS must be a non-negative integer, got '$GPUS'."
REPO_DIR=${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
read -r -a BATCH_ARGS <<<"$BATCH_SIZES"
[[ ${#BATCH_ARGS[@]} -gt 0 ]] || die "BATCH_SIZES is empty."
PY_CMD=(python scripts/benchmark_denoiser.py --hw "$HW" --steps "$STEPS" --batch-sizes "${BATCH_ARGS[@]}" --data "$DATA")

if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY_RUN: benchmark (in %s):' "$REPO_DIR"
    printf ' %q' "${PY_CMD[@]}"
    echo
    echo "DRY_RUN: BENCH lines are kept in the log (grep -h '^BENCH ' \$S/logs/bench-*.out)"
    exit 0
fi

[[ -f "$REPO_DIR/scripts/benchmark_denoiser.py" ]] || die "$REPO_DIR/scripts/benchmark_denoiser.py not found: submit from the repo checkout (or set REPO_DIR)."
[[ -d "$DATA" ]] || die "DATA directory $DATA not found."
cd "$REPO_DIR"
echo "$(date +%T) job ${SLURM_JOB_ID:-?} on $(hostname): hw $HW, gpus requested $GPUS, batch sizes ${BATCH_ARGS[*]}, steps $STEPS"
[[ "$GPUS" -le 1 ]] || echo "NOTE: benchmark_denoiser.py is single-process; it measures one of the $GPUS GPUs." >&2
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || true

# benchmark_denoiser.py prints one 'BENCH {json}' line per batch size to stdout; tee keeps them (and any warnings or
# tracebacks) in the log. Capture the real exit status: a failed benchmark must fail the job.
OUT=$(mktemp)
trap 'rm -f "$OUT"' EXIT
RC=0
"${PY_CMD[@]}" 2>&1 | tee "$OUT" || RC=$?
[[ "$RC" -eq 0 ]] || echo "ERROR: benchmark_denoiser.py exited with code $RC." >&2
if ! grep -q '^BENCH ' "$OUT"; then
    echo "ERROR: no BENCH lines were produced." >&2
    [[ "$RC" -ne 0 ]] || RC=1
fi
exit "$RC"
