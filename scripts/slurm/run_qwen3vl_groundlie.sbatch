#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /home/youruser/mavis-TFG
#SBATCH -t 0-20:00                    # 20h — anterior a 8h no fue suficiente (864/1722 vídeos)
#SBATCH -p pg1tfg12
#SBATCH --account=g1tfg12
#SBATCH --qos=q_pg1tfg12
#SBATCH --mem 64G
#SBATCH --gres=shard:11000            # 8B INT4 ~8-10 GB; margen extra
#SBATCH -o logs/%x_%u_%j.out
#SBATCH -e logs/%x_%u_%j.err
#SBATCH -J qwen8b_groundlie

mkdir -p logs

. /home/youruser/miniconda3/etc/profile.d/conda.sh
conda activate tfg2

echo "=== STARTING: Qwen3-VL-Embedding-8B on GroundLie360 ==="

python src/experiments/run_groundlie360_qwen3vl.py \
    --data-dir   data/GroundLie360 \
    --model-dir  src/models/Qwen3-VL-Embedding-8B \
    --qwen-repo  src/models/qwen3vl_embedding_repo \
    --output-dir experiments/GroundLie360/open_source/results/qwen3vl_8b \
    --index-json experiments/GroundLie360/groundlie_index_filtered.json \
    --bbox-json  experiments/GroundLie360/bbox_index.json \
    --max-frames 16 \
    --fps 1.0 \
    --quantize \
    --skip-ids 46453305

echo "=== FINISHED: Qwen3-VL-Embedding-8B on GroundLie360 ==="
