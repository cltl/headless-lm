#!/bin/bash
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --output=logs/publish.log

wdir=$HOME/headless-lm
# ckpt_dir=$wdir/checkpoints/mlm_vanilla/wikitext103-bpe_128
# ckpt="epoch=29-step=100000.ckpt"
# pt="epoch=29-step=100000.pt"

ckpt_dir=$wdir/checkpoints/gpt_vanilla_wiki103/wikitext103-bpe
ckpt="epoch=28-step=1500.ckpt"
pt="epoch=28-step=1500.pt"

# Loading modules
module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

# Prepare virtual environment
# virtualenv $wdir/.venv --system-site-packages
source "$wdir"/.venv/bin/activate
set -e

tmp_data_path="/gpfs/scratch1/nodespecific/gcn45/sarnoult.22719062/datasets"
dataset=wikitext103-bpe.hf

[[ ! -d "$tmp_data_path" ]] && mkdir -p "$tmp_data_path"
[[ ! -L "$tmp_data_path/$dataset" ]] && ln -s "$wdir"/datasets/"$dataset" "$tmp_data_path"/"$dataset"

# ln -s $wdir/datasets/wikitext103-bpe_128.hf \
#   /gpfs/scratch1/nodespecific/gcn11/sarnoult.22645100/datasets/wikitext103-bpe_128.hf

cd "$TMPDIR"
cp "$ckpt_dir/$ckpt" .

srun python "$wdir"/load_checkpoint.py "$ckpt" "$pt"
cp "$pt" "$ckpt_dir"
