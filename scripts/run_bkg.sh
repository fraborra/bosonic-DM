#!/bin/bash
# Run the background pipeline (dataset and Stan fit) on a NERSC compute node.
#
#   sbatch scripts/run_bkg.sh <group|all> [stage ...]
#
# `all` fits every detector group. Group names with spaces need quotes.
# Example: sbatch scripts/run_bkg.sh "ICPC group1" fit fit-plots
#
#SBATCH --job-name=bkg_fit
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=cpu
#SBATCH --time=04:00:00
#SBATCH --account=m2676
#SBATCH --mem=8G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

GROUP="${1:?usage: sbatch scripts/run_bkg.sh <group|all> [stage ...]}"
shift

ARGS=(--config configs/nersc.yaml)
if [[ "$GROUP" != "all" ]]; then
  ARGS+=(--group "$GROUP")
fi
if (($# > 0)); then
  ARGS+=(--stage "$@")
fi

# sbatch runs the job from the submission directory: submit from the repo root.
cd "${SLURM_SUBMIT_DIR:-.}"
# configs/nersc.yaml reads LEGEND_PRODUCTION_ROOT and BOSONIC_DM_SIM_ROOT.
# shellcheck source=/dev/null
source setup_env.sh
# One process: the batch script already runs on the compute node, no srun.
pixi run bosonic-dm background "${ARGS[@]}"
