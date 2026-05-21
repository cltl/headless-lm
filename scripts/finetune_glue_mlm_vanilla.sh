#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=8:00:00
#SBATCH --output=logs/glue_mlm_vanilla.log

wdir=$HOME/headless-lm

# Loading modules
module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

# Prepare virtual environment
# virtualenv $wdir/.venv --system-site-packages
source $wdir/.venv/bin/activate

cd $TMPDIR

run_name="glue_vmlm_100k"
model="CLTL-VUAmsterdam/BertMLM_wikitext"

srun python $wdir/glue_finetuning.py -c $wdir/configs/glue_mlm.json --model_id $model --run_name $run_name
