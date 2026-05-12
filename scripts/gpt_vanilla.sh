#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --cpus-per-task=18
#SBATCH --gpus=2
#SBATCH --time=30:00:00
#SBATCH --output=logs/gpt_vanilla.log

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
mkdir datasets
cp -r $wdir/datasets/wikitext103-bpe.hf datasets
srun python $wdir/gpt_headless.py -c $wdir/configs/gpt_vanilla_70m.json -j $wdir/configs/train_gpt_vanilla_wikitext103_bpe.json

ls -l
cp -r ckpts/* $wdir/checkpoints
