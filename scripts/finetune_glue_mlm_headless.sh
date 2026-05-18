#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --output=logs/glue_mlm_headless.log

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

ckpt_path="$wdir/checkpoints/mlm_headless/wikitext103/wikitext103-bpe_128/epoch=29-step=100000.ckpt"
ckpt="epoch=29-step=100000.ckpt"
run_name="hmlm_100k"
cp $ckpt_path $ckpt

srun python $wdir/glue_finetuning_ckpt.py -c $wdir/configs/glue_mlm_headless.json --ckpt_path $ckpt --run_name $run_name

ls -l
cp -r ckpts/* $wdir/checkpoints
