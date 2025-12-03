#!/bin/bash
#SBATCH --partition=gcn
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/mlm_headless.log

wdir=$HOME/headless-lm

# Loading modules
module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

# Prepare virtual environment
# virtualenv $wdir/.venv --system-site-packages
source $wdir/.venv/bin/activate

cd $TMPDIR
mkdir expdata
cd expdata
srun python $wdir/mlm_headless.py -c $wdir/configs/mlm_headless_test.json -j $wdir/configs/train_mlm_headless_wikitext_bpe_test.json

ls -l
cd ..
cp -r expdata $wdir
