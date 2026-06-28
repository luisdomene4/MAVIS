#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /home/youruser/mavis-TFG
#SBATCH -t 0-20:00                    # ~7971 scene clips over 1029 videos
#SBATCH -p pg1tfg12
#SBATCH --account=g1tfg12
#SBATCH --qos=q_pg1tfg12
#SBATCH --mem 64G
#SBATCH --gres=shard:11000
#SBATCH -o logs/%x_%u_%j.out
#SBATCH -e logs/%x_%u_%j.err
#SBATCH -J e3_qwen8b

mkdir -p logs
export PYTHONUNBUFFERED=1

. /home/youruser/miniconda3/etc/profile.d/conda.sh
conda activate tfg2

echo "=== STARTING: E3 scene embeddings — Qwen3-VL-Embedding-8B ==="

python scripts/preprocessing/build_e3_scene_embeddings.py \
    --model      qwen8b \
    --data-dir   data/GroundLie360 \
    --model-dir  src/models/Qwen3-VL-Embedding-8B \
    --qwen-repo  src/models/qwen3vl_embedding_repo \
    --max-frames 16 \
    --fps 1.0 \
    --quantize

echo "=== FINISHED: E3 scene embeddings — Qwen8B ==="
