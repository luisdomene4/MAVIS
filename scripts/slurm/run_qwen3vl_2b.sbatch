#!/bin/bash
#SBATCH --job-name=run_qwen3vl_2b
#SBATCH --output=logs/%x_%u_%j.out   # Output file
#SBATCH --error=logs/%x_%u_%j.err    # Error file
#SBATCH --cpus-per-task=4            # Whisper y procesamiento de DB se benefician de más CPUs
#SBATCH -N 1
#SBATCH -D /home/youruser/mavis-TFG
#SBATCH --time=12:00:00              # 12h es suficiente para ~1000 videos
#SBATCH -p pg1tfg12
#SBATCH --account=g1tfg12            
#SBATCH --qos=q_pg1tfg12             
#SBATCH --gres=shard:10000           # ~10 GB VRAM. El 2B cabe sin problema aquí (incluso sin cuantizar)
#SBATCH --mem=64G

mkdir -p logs

. /home/youruser/miniconda3/etc/profile.d/conda.sh
conda activate tfg2

# Evitar que Python guarde los prints en buffer (si hay OOM, lo veremos hasta el último print)
export PYTHONUNBUFFERED=1

echo "=== STARTING FULL RUN: Qwen3-VL-Embedding-2B ==="

python src/experiments/run_qwen3vl.py \
    --data-dir   data/FakeVV_testset \
    --model-dir  src/models/Qwen3-VL-Embedding-2B \
    --qwen-repo  src/models/qwen3vl_embedding_repo \
    --output-dir experiments/open_source/results/qwen3vl_2b \
    --max-frames 16 \
    --fps        1.0 

echo "=== FINISHED FULL RUN ==="
