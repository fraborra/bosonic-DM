#!/bin/bash
# Run the simulation pipeline on a NERSC compute node.
#
#   sbatch scripts/run_sim.sh <interaction> [stage ...]
#
# Example: sbatch scripts/run_sim.sh axio-electric efficiencies
#
#SBATCH --job-name=sim_eff
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=cpu
#SBATCH --time=02:00:00
#SBATCH --account=m2676
#SBATCH --mem-per-cpu=2G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

INTERACTION="${1:?usage: sbatch scripts/run_sim.sh <interaction> [stage ...]}"
shift

ARGS=(--config configs/nersc.yaml --interaction "$INTERACTION")
if (($# > 0)); then
  ARGS+=(--stage "$@")
fi

# sbatch runs the job from the submission directory: submit from the repo root.
cd "${SLURM_SUBMIT_DIR:-.}"
# configs/nersc.yaml reads LEGEND_PRODUCTION_ROOT and BOSONIC_DM_SIM_ROOT.
# shellcheck source=/dev/null
source setup_env.sh
# One process: the batch script already runs on the compute node, no srun.
pixi run bosonic-dm simulation "${ARGS[@]}"
