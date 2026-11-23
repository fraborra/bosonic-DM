#!/bin/bash
# Run the background pipeline (dataset and Stan fit) on a NERSC compute node.
#
#   sbatch scripts/run_bkg.sh <group|all> [--overwrite] [stage ...]
#
# `all` fits every detector group. Group names with spaces need quotes.
# `--overwrite` regenerates existing products (it can appear anywhere after
# the group).
# Example: sbatch scripts/run_bkg.sh "ICPC group1" --overwrite fit fit-plots
#
#SBATCH --job-name=bkg_fit
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=cpu
#SBATCH --time=01:40:00
#SBATCH --account=m2676
#SBATCH --mem-per-cpu=1G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

GROUP="${1:?usage: sbatch scripts/run_bkg.sh <group|all> [stage ...]}"
shift

ARGS=(--config configs/nersc.yaml)
STAGES=()
for arg in "$@"; do
  if [[ "$arg" == "--overwrite" ]]; then
    ARGS+=(--overwrite)
  else
    STAGES+=("$arg")
  fi
done
if [[ "$GROUP" != "all" ]]; then
  ARGS+=(--group "$GROUP")
fi
if ((${#STAGES[@]} > 0)); then
  ARGS+=(--stage "${STAGES[@]}")
fi

# sbatch runs the job from the submission directory: submit from the repo root.
cd "${SLURM_SUBMIT_DIR:-.}"
# configs/nersc.yaml reads LEGEND_PRODUCTION_ROOT and BOSONIC_DM_SIM_ROOT.
# shellcheck source=/dev/null
source setup_env.sh
# One process: the batch script already runs on the compute node, no srun.
pixi run bosonic-dm background "${ARGS[@]}"
