#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /home/youruser/mavis-TFG
#SBATCH -t 0-12:00
#SBATCH -p pg1tfg12
#SBATCH --account=g1tfg12
#SBATCH --qos=q_pg1tfg12
#SBATCH --mem 64G
#SBATCH --gres=shard:11000            # 2B INT4 ~4-5 GB
#SBATCH -o logs/%x_%u_%j.out
#SBATCH -e logs/%x_%u_%j.err
#SBATCH -J qwen2b_m3a

mkdir -p logs
export PYTHONUNBUFFERED=1

. /home/youruser/miniconda3/etc/profile.d/conda.sh
conda activate tfg2

INDEX="${M3A_INDEX:-experiments/M3A/m3a_index_2500.json}"
echo "=== STARTING: Qwen3-VL-Embedding-2B on M3A ($INDEX) ==="

# Smoke test: add --limit 5
python src/experiments/run_m3a_qwen3vl.py \
    --data-dir   data/M3A \
    --model-dir  src/models/Qwen3-VL-Embedding-2B \
    --qwen-repo  src/models/qwen3vl_embedding_repo \
    --output-dir experiments/M3A/open_source/results/qwen3vl_2b \
    --index-json "$INDEX" \
    --max-frames 16 \
    --fps 1.0 \
    --quantize

echo "=== FINISHED: Qwen3-VL-Embedding-2B on M3A ==="
