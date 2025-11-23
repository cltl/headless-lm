#!/bin/bash
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --output=logs/setup_venv.log

wdir=$HOME/headless-lm

# Loading modules
module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

# Prepare virtual environment
virtualenv $wdir/.venv --system-site-packages
source $wdir/.venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip list
pip install -r $wdir/requirements.txt

