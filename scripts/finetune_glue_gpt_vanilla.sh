#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --output=logs/glue_gpt_vanilla.log

wdir=$HOME/headless-lm

# Loading modules
module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

# Prepare virtual environment
# virtualenv $wdir/.venv --system-site-packages
source $wdir/.venv/bin/activate

cd $TMPDIR

run_name="vgpt_1k5"
model="CLTL-VUAmsterdam/GPT-pythia-70m-wikitext"
tokenizer="CLTL/wikitext103-BPE-50k"

srun python $wdir/glue_finetuning.py -c $wdir/configs/glue_gpt.json --model_id $model --run_name $run_name -t $tokenizer
