#!/bin/bash
# ------------------------------------------------------------------------------
# Environment Setup for Bosonic DM Analysis
# ------------------------------------------------------------------------------
# Source this file before running the pipeline or starting your notebook server:
#   source setup_env.sh
# ------------------------------------------------------------------------------

# 1. LEGEND Production Metadata
# The root directory containing the LEGEND metadata repository
export LEGEND_PRODUCTION_ROOT="/global/cfs/projectdirs/m2676/data/lngs/l200/public/prodenv/prod-blind/ref/"

# 2. Bosonic DM Simulations Root
# The root directory containing your generated LH5 Tier CVT and STP files
export BOSONIC_DM_SIM_ROOT="/pscratch/sd/b/borrfran/sim-v1.1.0-20260401/"

echo "Environment variables loaded:"
echo " - LEGEND_PRODUCTION_ROOT = ${LEGEND_PRODUCTION_ROOT}"
echo " - BOSONIC_DM_SIM_ROOT    = ${BOSONIC_DM_SIM_ROOT}"
