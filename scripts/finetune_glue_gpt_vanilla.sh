#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --output=logs/glue_gpt_vanilla.log

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

ckpt_path="$wdir/checkpoints/gpt_vanilla_wiki103/wikitext103-bpe/epoch=28-step=1500.ckpt"
ckpt="vmlm_epoch=28-step=1500.ckpt"
run_name="vgpt_1500"
cp $ckpt_path $ckpt

srun python $wdir/glue_finetuning_ckpt.py -c $wdir/configs/glue_gpt.json --ckpt_path $ckpt --run_name $run_name

ls -l
# cp -r ckpts/* $wdir/checkpoints
