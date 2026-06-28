#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /home/youruser/mavis-TFG
#SBATCH -t 0-01:00                    # text-only, ~40 embeds; model load dominates
#SBATCH -p pg1tfg12
#SBATCH --account=g1tfg12
#SBATCH --qos=q_pg1tfg12
#SBATCH --mem 64G
#SBATCH --gres=shard:11000
#SBATCH -o logs/%x_%u_%j.out
#SBATCH -e logs/%x_%u_%j.err
#SBATCH -J e2_qwen2b

mkdir -p logs
export PYTHONUNBUFFERED=1

. /home/youruser/miniconda3/etc/profile.d/conda.sh
conda activate tfg2

echo "=== STARTING: E2 windows — Qwen3-VL-Embedding-2B ==="

python scripts/preprocessing/build_e2_windows.py \
    --model      qwen2b \
    --model-dir  src/models/Qwen3-VL-Embedding-2B \
    --qwen-repo  src/models/qwen3vl_embedding_repo \
    --quantize

echo "=== FINISHED: E2 windows — Qwen2B ==="