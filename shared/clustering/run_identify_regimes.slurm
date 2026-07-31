#!/bin/bash
#SBATCH --job-name=regimes_hh_gmm
#SBATCH --partition=std
#SBATCH --mail-type=NONE
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/regimes_%j.out
#SBATCH --error=logs/regimes_%j.err

# Half-hourly GMM regime clustering for IDW, BSS, RFSI
# Writes assignments, features, medoids into each method's clusters folder.

set -euo pipefail

source ~/venv_Master/bin/activate

cd /media/data/Stefan/Masterarbeit/Daten/CODE/shared/clustering
mkdir -p logs

echo "=============================================================="
echo "Regime clustering | half_hourly | gmm | k=[3..20] | n_medoids=20"
echo "methods: IDW BSS RFSI"
echo "Started: $(date)"
echo "=============================================================="

python identify_regimes.py \
  --method IDW BSS RFSI \
  --resolution half_hourly \
  --cluster-method gmm \
  --k-min 3 \
  --k-max 20 \
  --n-medoids 20 \
  --pca-variance 0.95

echo "Finished: $(date)"
