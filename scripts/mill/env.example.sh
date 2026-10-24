# Copy to ~/.dcs2_env on the Mill and `source` it (see #3, docs/MILL.md). It is sourced by every sbatch script, so keep it
# quiet and idempotent.
export S=/share/ceph/scratch/$USER/dcs2
# replace with the type strings found via sinfo in #3
export H100_GRES=gpu:h100:1
export V100_GRES=gpu:v100:1
conda activate dcs2 2>/dev/null || true
