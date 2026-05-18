#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --output=logs/glue_gpt_headless.log

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

ckpt_path="$wdir/checkpoints/gpt_headless_clm/wikitext103/epoch=0-step=1500.ckpt"
ckpt="hgpt_epoch=0-step=1500.ckpt"
run_name="hgpt_1500"
cp $ckpt_path $ckpt

srun python $wdir/glue_finetuning_ckpt.py -c $wdir/configs/glue_gpt.json --ckpt_path $ckpt --run_name $run_name

ls -l
cp -r ckpts/* $wdir/checkpoints
