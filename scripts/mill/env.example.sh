# Copy to ~/.dcs2_env on the Mill and `source` it (see #3, docs/MILL.md). It is sourced by every sbatch script, so keep it
# quiet and idempotent.
export S=/share/ceph/scratch/$USER/dcs2
# replace with the type strings found via sinfo in #3
export H100_GRES=gpu:h100:1
export V100_GRES=gpu:v100:1
# The Mill ships conda as a module, and `conda activate` needs the shell hook in shells where `conda init` never ran
# (batch jobs included).
module load anaconda >/dev/null 2>&1 || true
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate dcs2 2>/dev/null || true
