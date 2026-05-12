#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=30:00:00
#SBATCH --output=logs/mlm_headless_2.log

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
# cp -r $wdir/datasets/wikitext2-bpe.hf datasets
# srun python $wdir/mlm_headless.py -c $wdir/configs/mlm_headless_test.json -j $wdir/configs/train_mlm_headless_wikitext_bpe_test.json
cp -r $wdir/datasets/wikitext103-bpe_128.hf datasets
srun python $wdir/mlm_headless.py -c $wdir/configs/mlm_headless.json -j $wdir/configs/train_mlm_headless_wikitext103_bpe.json

ls -l
cp -r ckpts/* $wdir/checkpoints
[[ -d wandb ]] && cp -r wandb $wdir
