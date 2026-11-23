#!/bin/bash
# Stan versus grid-posterior limits on the same toys, as a SLURM job array.
#
# Compile the model once before submitting (the jobs would otherwise race):
#   PYTHONPATH=src pixi run python3 -c "from cmdstanpy import CmdStanModel; from bosonic_dm.fitting.signal_fit import SIGNAL_STAN_FILE; CmdStanModel(stan_file=str(SIGNAL_STAN_FILE))"
#
# Array (the masses are split across the tasks):
#   sbatch --array=0-3 scripts/submit_stan_comparison.sh \
#     --config configs/nersc.yaml \
#     --grid-per-toy data/v5/sensitivity/full/limits_per_toy.parquet \
#     --outdir data/v5/sensitivity/stan_check
#
# Optional final comparison of everything written (after all tasks end):
#   sbatch scripts/submit_stan_comparison.sh --config configs/nersc.yaml \
#     --outdir data/v5/sensitivity/stan_check --compare-only
#
# Submit from the repository root: the SLURM log goes to _out/, which must
# exist (it is kept in git by _out/.gitkeep).
#
#SBATCH --job-name=stan_check
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=cpu
#SBATCH --time=04:00:00
#SBATCH --account=m2676
#SBATCH --mem=8G
#SBATCH --output=_out/slurm-%x-%A_%a.out

set -euo pipefail

# sbatch runs the job from the submission directory: submit from the repo root.
cd "${SLURM_SUBMIT_DIR:-.}"
# shellcheck source=/dev/null
source setup_env.sh
PYTHONPATH=src pixi run python3 scripts/stan_grid_comparison.py "$@"
