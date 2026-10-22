#!/bin/bash
#SBATCH -p gpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o /share/ceph/scratch/%u/dcs2/logs/%x-%j.out
# NOTE: #SBATCH lines do not expand $S, so the -o path spells out the scratch root (S=/share/ceph/scratch/$USER/dcs2).
#
# Mill denoiser benchmark (#7, used by #8). Environment: HW, DATA, BATCH_SIZES, STEPS (and GPUS for the DDP probe).
# Submit with: sbatch --gres=$H100_GRES -J bench-h100 --export=ALL,HW=h100,DATA=...,BATCH_SIZES="32 64 128",STEPS=200 \
#                     scripts/mill/benchmark.sbatch
#
# TODO(#7): source ~/.dcs2_env.
# TODO(#7): python scripts/benchmark_denoiser.py --hw $HW --steps $STEPS --batch-sizes $BATCH_SIZES --data $DATA
# TODO(#7): grep the '^BENCH ' lines into the log.

echo "TODO(#7): not implemented" >&2
exit 2
