#!/bin/bash
#SBATCH -p gpu
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -o /share/ceph/scratch/%u/dcs2/logs/%x-%j.out
# NOTE: #SBATCH lines do not expand $S, so the -o path spells out the scratch root (S=/share/ceph/scratch/$USER/dcs2).
#
# Mill training job (#7). Environment: RUN_NAME, DATASET, HW, RESUBMITS_LEFT (default 3), EXTRA (Hydra overrides).
# Submit with: sbatch --gres=$H100_GRES -J $RUN_NAME --export=ALL,RUN_NAME=...,DATASET=...,HW=h100 scripts/mill/train.sbatch
#
# TODO(#7): source ~/.dcs2_env; set RESUBMITS_LEFT default.
# TODO(#7): if RESUBMITS_LEFT>0, submit the successor with
#           sbatch --parsable --dependency=afterany:$SLURM_JOB_ID (RESUBMITS_LEFT-1) and print its job id.
# TODO(#7): rsync $S/datasets/$DATASET to /local/scratch/$USER/$SLURM_JOB_ID/; trap 'rm -rf ...' EXIT.
# TODO(#7): run in the background:
#           python src/main.py hydra.run.dir=$S/runs/$RUN_NAME common.resume=auto hw=$HW \
#               env.path_data_low_res=<staged> $EXTRA
#           with trap 'kill -USR1 $PID' USR1, and wait on $PID correctly across the signal.
# TODO(#7): exit handling: checkpoints/DONE -> scancel the successor; exit code 3 (preempted) -> leave the successor
#           queued; any other non-zero exit -> scancel the successor and print a clear error.

echo "TODO(#7): not implemented" >&2
exit 2
