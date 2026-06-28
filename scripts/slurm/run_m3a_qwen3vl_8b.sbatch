#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /home/youruser/mavis-TFG
#SBATCH -t 0-20:00
#SBATCH -p pg1tfg12
#SBATCH --account=g1tfg12
#SBATCH --qos=q_pg1tfg12
#SBATCH --mem 64G
#SBATCH --gres=shard:11000            # 8B INT4 ~8-10 GB
#SBATCH -o logs/%x_%u_%j.out
#SBATCH -e logs/%x_%u_%j.err
#SBATCH -J qwen8b_m3a

mkdir -p logs
export PYTHONUNBUFFERED=1

. /home/youruser/miniconda3/etc/profile.d/conda.sh
conda activate tfg2

INDEX="${M3A_INDEX:-experiments/M3A/m3a_index_2500.json}"
echo "=== STARTING: Qwen3-VL-Embedding-8B on M3A ($INDEX) ==="

# --qwen-db: reutiliza los transcripts del 2B → NO carga Whisper (más VRAM para el 8B).
python src/experiments/run_m3a_qwen3vl.py \
    --data-dir   data/M3A \
    --model-dir  src/models/Qwen3-VL-Embedding-8B \
    --qwen-repo  src/models/qwen3vl_embedding_repo \
    --output-dir experiments/M3A/open_source/results/qwen3vl_8b \
    --index-json "$INDEX" \
    --qwen-db    experiments/M3A/open_source/results/qwen3vl_2b/qwen3vl_cache.db \
    --max-frames 16 \
    --fps 1.0 \
    --quantize

echo "=== FINISHED: Qwen3-VL-Embedding-8B on M3A ==="
