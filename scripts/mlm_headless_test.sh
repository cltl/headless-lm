#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=0:20:00
#SBATCH --output=logs/mlm_headless_test.log

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
[[ ! -d datasets ]] && mkdir datasets
cp -r $wdir/datasets/wikitext2-bpe.hf datasets

ckptdir="checkpoints/mlm_headless/wikitext2-bpe"
[[ ! -d checkpoints ]] && mkdir -p checkpoints
cp "$wdir/$ckptdir/epoch=2-step=150.ckpt" checkpoints
srun python $wdir/mlm_headless.py -c $wdir/configs/mlm_headless_test.json -j $wdir/configs/train_mlm_headless_wikitext_bpe_cpt_test.json

ls -l
ls -l checkpoints
cp -r "$ckptdir"_from_epoch=2-step=150 $wdir
