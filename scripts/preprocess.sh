#!/bin/bash
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/preprocess.log

wdir=$HOME/headless-lm

# Loading modules
module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

# Prepare virtual environment
# virtualenv $wdir/.venv --system-site-packages
source $wdir/.venv/bin/activate
# pip install markupsafe==2.0.1

set -e

cd $TMPDIR

echo "Creating BPE tokenizer"
srun python $wdir/preprocess.py --config $wdir/configs/preprocess_wikitext_bpe.json
echo "Creating WP tokenizer"
srun python $wdir/preprocess.py --config $wdir/configs/preprocess_wikitext_wp.json

cp -r datasets $wdir
