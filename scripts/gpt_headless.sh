#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --output=logs/gpt_headless.log

wdir=$HOME/headless-lm
[[ ! -d $wdir/checkpoints ]] && mkdir $wdir/checkpoints

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
mkdir datasets
cp -r $wdir/datasets/wikitext2-bpe.hf datasets
srun python $wdir/gpt_headless.py -c $wdir/configs/gpt_headless_test.json -j $wdir/configs/train_gpt_headless_wikitext_bpe_test.json

ls -l
cd ..
cp -r expdata/ckpts/* $wdir/checkpoints
