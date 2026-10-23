#!/bin/bash
#SBATCH -p gpu
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -o /share/ceph/scratch/%u/dcs2/logs/%x-%j.out
# NOTE: #SBATCH lines do not expand $S, so the -o path spells out the scratch root (S=/share/ceph/scratch/$USER/dcs2).
#
# Mill training job (#7). Every training job is this one script. Submit from the repo checkout (the job runs there):
#   sbatch --gres=$H100_GRES -J $RUN_NAME --export=ALL,RUN_NAME=...,DATASET=...,HW=h100 scripts/mill/train.sbatch
#
# Environment:
#   RUN_NAME        required. Run dir is $S/runs/$RUN_NAME (checkpoints go straight to scratch, never /local/scratch).
#   DATASET         required. $S/datasets/$DATASET (with low_res/{train,test}) is rsynced to node-local scratch.
#   HW              required. config/hw profile: h100, v100 or rtx3060.
#   RESUBMITS_LEFT  optional, default 3. If > 0, the successor job is submitted at job START (so a hard kill still
#                   resumes) with RESUBMITS_LEFT-1 and dependency afterany:$SLURM_JOB_ID.
#   EXTRA           optional. Whitespace-separated Hydra overrides appended to the trainer command.
#   GRES            optional. --gres for the successor. Default: $H100_GRES for HW=h100, $V100_GRES for HW=v100
#                   (from ~/.dcs2_env). Required for other HW values. Pass the same value as the original --gres.
#   TIME_LIMIT      optional. --time for the successor. Default: this job's own limit (from squeue), else the
#                   #SBATCH default above.
#   LOCAL_SCRATCH_ROOT  optional, default /local/scratch. Staged data lives in $LOCAL_SCRATCH_ROOT/$USER/$SLURM_JOB_ID.
#   REPO_DIR        optional, default $SLURM_SUBMIT_DIR (else $PWD). Must contain src/main.py.
#   DRY_RUN=1       print the planned sbatch / rsync / python commands and exit without running anything.
#
# Trainer contract (#6): exit 0 = finished (or checkpoints/DONE already existed), exit 3 = preempted (state.pt and
# checkpoints/PREEMPTED written; resume with the successor), anything else = crash. SLURM sends SIGUSR1 to this batch
# shell 600 s before the time limit (--signal=B:); it is forwarded to Python, which saves at the next optimizer step.
set -euo pipefail

die() {
    echo "train.sbatch: ERROR: $*" >&2
    exit 2
}

# ---- 1. Environment -------------------------------------------------------------------------------------------------
DRY_RUN=${DRY_RUN:-0}
if [[ -f "$HOME/.dcs2_env" ]]; then
    set +u
    # shellcheck disable=SC1091
    source "$HOME/.dcs2_env"
    set -u
elif [[ "$DRY_RUN" != 1 ]]; then
    die "$HOME/.dcs2_env not found. Copy scripts/mill/env.example.sh there and edit it (docs/MILL.md)."
fi

for var in S RUN_NAME DATASET HW; do
    [[ -n "${!var:-}" ]] || die "$var is not set (required: S via ~/.dcs2_env; RUN_NAME, DATASET, HW via sbatch --export=ALL,...)."
done
RESUBMITS_LEFT=${RESUBMITS_LEFT:-3}
[[ "$RESUBMITS_LEFT" =~ ^[0-9]+$ ]] || die "RESUBMITS_LEFT must be a non-negative integer, got '$RESUBMITS_LEFT'."
EXTRA=${EXTRA:-}
if [[ "$DRY_RUN" == 1 ]]; then
    JOB_ID=${SLURM_JOB_ID:-dryrun}
else
    JOB_ID=${SLURM_JOB_ID:-}
    [[ -n "$JOB_ID" ]] || die "SLURM_JOB_ID is not set: submit with sbatch (or use DRY_RUN=1 to preview)."
fi
JOB_NAME=${SLURM_JOB_NAME:-$RUN_NAME}
REPO_DIR=${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
[[ "$DRY_RUN" == 1 || -f "$REPO_DIR/src/main.py" ]] || die "$REPO_DIR/src/main.py not found: submit from the repo checkout (or set REPO_DIR)."
LOCAL_SCRATCH_ROOT=${LOCAL_SCRATCH_ROOT:-/local/scratch}

if [[ -z "${GRES:-}" ]]; then
    case "$HW" in
        h100) GRES=${H100_GRES:-} ;;
        v100) GRES=${V100_GRES:-} ;;
        *) GRES= ;;
    esac
fi
if [[ -z "$GRES" && "$RESUBMITS_LEFT" -gt 0 ]]; then
    die "cannot determine --gres for the successor job: set GRES (or H100_GRES / V100_GRES in ~/.dcs2_env), or RESUBMITS_LEFT=0."
fi
if [[ -z "${TIME_LIMIT:-}" && "$DRY_RUN" != 1 ]] && command -v squeue >/dev/null 2>&1; then
    TIME_LIMIT=$(squeue -h -j "$JOB_ID" -o %l 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)
fi

RUN_DIR=$S/runs/$RUN_NAME
STAGE_DIR=$LOCAL_SCRATCH_ROOT/${USER:-$(id -un)}/$JOB_ID
LOW_RES=$STAGE_DIR/$DATASET/low_res

# ---- 2. Planned commands --------------------------------------------------------------------------------------------
SBATCH_CMD=(sbatch --parsable "--dependency=afterany:$JOB_ID" -p gpu "--gres=$GRES" -J "$JOB_NAME"
    "--export=ALL,RESUBMITS_LEFT=$((RESUBMITS_LEFT - 1))")
[[ -z "${TIME_LIMIT:-}" ]] || SBATCH_CMD+=("--time=$TIME_LIMIT")
SBATCH_CMD+=("$REPO_DIR/scripts/mill/train.sbatch")
RSYNC_CMD=(rsync -a "$S/datasets/$DATASET/" "$STAGE_DIR/$DATASET/")
read -r -a EXTRA_ARGS <<<"$EXTRA" # split on whitespace without globbing (Hydra overrides contain [ ] , *)
PY_CMD=(python src/main.py "hydra.run.dir=$RUN_DIR" common.resume=auto "hw=$HW" "env.path_data_low_res=$LOW_RES"
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})

# Print "DRY_RUN: <label> <command>", shell-quoting only the words that need it (so the output stays copy-pasteable).
show_cmd() {
    local label=$1 word
    shift
    printf 'DRY_RUN: %s' "$label"
    for word in "$@"; do
        if [[ "$word" =~ ^[A-Za-z0-9_@%+=:,./-]+$ ]]; then printf ' %s' "$word"; else printf ' %q' "$word"; fi
    done
    echo
}

if [[ "$DRY_RUN" == 1 ]]; then
    echo "DRY_RUN: job $JOB_ID, run $RUN_NAME, dataset $DATASET, hw $HW, resubmits left $RESUBMITS_LEFT"
    if [[ "$RESUBMITS_LEFT" -gt 0 ]]; then
        show_cmd "successor:" "${SBATCH_CMD[@]}"
    else
        echo "DRY_RUN: successor: none (RESUBMITS_LEFT=0)"
    fi
    show_cmd "stage:" mkdir -p "$STAGE_DIR"
    show_cmd "stage:" "${RSYNC_CMD[@]}"
    show_cmd "train (in $REPO_DIR, background, USR1 forwarded):" "${PY_CMD[@]}"
    echo "DRY_RUN: on exit: rm -rf $STAGE_DIR"
    echo "DRY_RUN: exit 0 or $RUN_DIR/checkpoints/DONE -> scancel successor; exit 3 -> keep successor; else cancel it"
    exit 0
fi

# ---- 3. Cleanup and signal handling ---------------------------------------------------------------------------------
# PHASE decides what the EXIT trap does with the successor: it is cancelled only when training is finished ("done"), has
# crashed ("crash") or failed during setup ("setup"). A job killed while "running" (wall-time, node failure) or
# "preempted" leaves the successor queued so the chain resumes.
PHASE=setup
SUCC_ID=
PID=
USR1_EARLY=0

cleanup() {
    local rc=$?
    trap - EXIT
    rm -rf "$STAGE_DIR"
    case "$PHASE" in
        setup | done | crash)
            if [[ -n "$SUCC_ID" ]]; then
                echo "Cancelling successor job $SUCC_ID ($PHASE)."
                scancel "$SUCC_ID" || echo "WARNING: scancel $SUCC_ID failed; cancel it by hand." >&2
            fi
            ;;
    esac
    exit "$rc"
}
forward_usr1() {
    if [[ -n "$PID" ]]; then
        echo "$(date +%T) SIGUSR1 received: forwarding to trainer (pid $PID)."
        kill -USR1 "$PID" 2>/dev/null || true
    else
        USR1_EARLY=1
    fi
}
on_term() {
    echo "$(date +%T) SIGTERM received: stopping the trainer." >&2
    [[ -z "$PID" ]] || kill -TERM "$PID" 2>/dev/null || true
    exit 143
}
trap cleanup EXIT
trap forward_usr1 USR1
trap on_term TERM

echo "$(date +%T) job $JOB_ID on $(hostname): run $RUN_NAME, dataset $DATASET, hw $HW, resubmits left $RESUBMITS_LEFT"

# ---- 4. Successor (submitted at start so that hard kills still resume) ----------------------------------------------
if [[ "$RESUBMITS_LEFT" -gt 0 ]]; then
    if SUCC_ID=$(cd "$REPO_DIR" && "${SBATCH_CMD[@]}"); then
        SUCC_ID=${SUCC_ID%%;*} # --parsable prints "<id>[;<cluster>]"
        echo "Successor job: $SUCC_ID (afterany:$JOB_ID, resubmits left: $((RESUBMITS_LEFT - 1)))"
    else
        SUCC_ID=
        echo "WARNING: could not submit the successor job; this run will not chain." >&2
    fi
else
    echo "RESUBMITS_LEFT=0: no successor job."
fi

# ---- 5. Stage the read-only dataset to node-local scratch -----------------------------------------------------------
[[ -d "$S/datasets/$DATASET" ]] || die "dataset $S/datasets/$DATASET not found."
mkdir -p "$RUN_DIR" "$STAGE_DIR"
echo "$(date +%T) staging $S/datasets/$DATASET -> $STAGE_DIR/$DATASET"
"${RSYNC_CMD[@]}"
[[ -d "$LOW_RES" ]] || die "staged dataset has no low_res/ directory ($LOW_RES); expected \$S/datasets/$DATASET/low_res/{train,test}."
echo "$(date +%T) staging done"

# ---- 6. Train -------------------------------------------------------------------------------------------------------
if [[ "$USR1_EARLY" == 1 ]]; then
    echo "SIGUSR1 arrived before training started: not launching; the successor resumes."
    PHASE=preempted
    exit 3
fi
cd "$REPO_DIR"
MARKER=$STAGE_DIR/.launched
touch "$MARKER"
printf 'Launching: %q ' "${PY_CMD[@]}"
echo
"${PY_CMD[@]}" &
PID=$!
PHASE=running

# `wait` returns early (status > 128) when a trapped signal such as USR1 arrives, with the child still running. Loop
# until the child is really gone, then take its real exit status from the final `wait`.
RC=0
while :; do
    RC=0
    wait "$PID" || RC=$?
    if kill -0 "$PID" 2>/dev/null; then
        continue # interrupted by a trapped signal; the child is still alive (or an unreaped zombie)
    fi
    break
done
PID=
CKPT_DIR=$RUN_DIR/checkpoints

# ---- 7. Exit handling -----------------------------------------------------------------------------------------------
if [[ "$RC" -eq 0 ]]; then
    PHASE=done
    if [[ -f "$CKPT_DIR/DONE" ]]; then
        echo "$(date +%T) training finished ($CKPT_DIR/DONE)."
    else
        echo "WARNING: trainer exited 0 but $CKPT_DIR/DONE is missing; treating training as finished." >&2
    fi
    exit 0
fi
# PREEMPTED is a secondary signal only: it counts if written by this job (newer than the launch marker).
if [[ "$RC" -eq 3 || "$CKPT_DIR/PREEMPTED" -nt "$MARKER" ]]; then
    PHASE=preempted
    echo "$(date +%T) trainer preempted (exit $RC, state saved in $CKPT_DIR); the successor job resumes the run."
    [[ -n "$SUCC_ID" ]] || echo "No successor is queued (RESUBMITS_LEFT=0): resubmit by hand to continue." >&2
    exit 3
fi
PHASE=crash
if [[ -f "$CKPT_DIR/DONE" ]]; then
    echo "WARNING: trainer exited $RC after writing $CKPT_DIR/DONE; training itself is complete." >&2
fi
echo "ERROR: trainer crashed with exit code $RC (not a preemption). Successor cancelled to avoid a crash loop;" \
    "see the traceback above in this log, fix the cause, and resubmit." >&2
exit "$RC"
