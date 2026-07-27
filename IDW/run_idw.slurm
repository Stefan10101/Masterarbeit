#!/bin/bash
#SBATCH --job-name=idw_maps
#SBATCH --mail-type=NONE
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=logs/idw_%j.out
#SBATCH --error=logs/idw_%j.err

# activate venv
source ~/venv_Master/bin/activate

# go to the exact script directory
cd /media/data/Stefan/Masterarbeit/Daten/CODE/IDW

# create log directory if needed
mkdir -p logs

# run
python produce_idw_maps.py
