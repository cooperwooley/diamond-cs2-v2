# Copy to ~/.dcs2_env on the Mill and `source` it (see #3, docs/MILL.md). It is sourced by every sbatch script, so keep it
# quiet and idempotent.
export S=/share/ceph/scratch/$USER/dcs2
# GRES strings and partitions as reported by `sinfo -a -N -o "%N %P %G"`. There is no `gpu` partition. `requeue` is the
# free tier (1-day limit, jobs can be preempted; it includes the H100 and V100 nodes). std-h100 / std-v100 are paid
# (MOU + MOCode) and not available to the `general` account, so they are not used. Submit with -p $H100_PARTITION.
export H100_GRES=gpu:H100:1
export V100_GRES=gpu:V100-SXM2-32GB:1
export H100_PARTITION=requeue
export V100_PARTITION=requeue
# Prefer a Miniforge install in ~/miniforge3, else the anaconda module. `conda activate` needs the shell hook in shells
# where `conda init` never ran (batch jobs included).
if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then source "$HOME/miniforge3/etc/profile.d/conda.sh"
else module load anaconda >/dev/null 2>&1 || true; eval "$(conda shell.bash hook 2>/dev/null)" || true; fi
conda activate dcs2 2>/dev/null || true
