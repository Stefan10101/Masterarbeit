#!/bin/bash
#SBATCH --job-name=train_cluster_params
#SBATCH --partition=std
#SBATCH --mail-type=NONE
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_params_%j.out
#SBATCH --error=logs/train_params_%j.err

# Train free parameters on cluster medoids for IDW, RFSI, BSS.
# Requires identify_regimes (half_hourly / gmm) to have finished first.

set -euo pipefail

source ~/venv_Master/bin/activate

ROOT=/media/data/Stefan/Masterarbeit/Daten/CODE
mkdir -p "$ROOT/shared/clustering/logs"

RESOLUTION=half_hourly
CLUSTER_METHOD=gmm
N_JOBS=-1

echo "=============================================================="
echo "Train cluster parameters | ${RESOLUTION} | ${CLUSTER_METHOD}"
echo "Started: $(date)"
echo "=============================================================="

# --- IDW ---
echo ""
echo ">>> IDW"
cd "$ROOT/IDW"
python train_cluster_params.py \
  --resolution "$RESOLUTION" \
  --cluster-method "$CLUSTER_METHOD" \
  --n-jobs "$N_JOBS"

# --- RFSI ---
echo ""
echo ">>> RFSI"
cd "$ROOT/RFSI"
python train_cluster_params.py \
  --resolution "$RESOLUTION" \
  --cluster-method "$CLUSTER_METHOD" \
  --n-jobs "$N_JOBS"

# --- BSS ---
echo ""
echo ">>> BSS"
cd "$ROOT/BSS"
python train_cluster_params.py \
  --resolution "$RESOLUTION" \
  --cluster-method "$CLUSTER_METHOD"

echo ""
echo "All training finished: $(date)"
