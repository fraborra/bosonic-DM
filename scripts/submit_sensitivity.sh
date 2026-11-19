#!/bin/bash
# Run the expected-sensitivity study on a NERSC compute node.
#
#   sbatch scripts/submit_sensitivity.sh [expected_sensitivity.py arguments]
#
# Example (smoke run):
#   sbatch scripts/submit_sensitivity.sh --config configs/nersc.yaml \
#     --masses 300,500,800 --n-toys 200 --outdir data/v4/sensitivity/smoke
#
#SBATCH --job-name=sensitivity
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --constraint=cpu
#SBATCH --time=01:00:00
#SBATCH --account=m2676
#SBATCH --mem=8G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# sbatch runs the job from the submission directory: submit from the repo root.
cd "${SLURM_SUBMIT_DIR:-.}"
# shellcheck source=/dev/null
source setup_env.sh
PYTHONPATH=src pixi run python3 scripts/expected_sensitivity.py "$@"
