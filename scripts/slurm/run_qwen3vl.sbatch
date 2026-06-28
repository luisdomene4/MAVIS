#!/bin/bash
#SBATCH -n 4                          # Number of cores
#SBATCH -N 1                          # All cores on one node
#SBATCH -D /home/youruser/mavis-TFG   # Working directory
#SBATCH -t 0-04:00                    # Runtime: 4 hours
#SBATCH -p pg1tfg12                   # Partition
#SBATCH --account=g1tfg12             # Account
#SBATCH --qos=q_pg1tfg12             # QoS
#SBATCH --mem 64G                   # 32GB memory
#SBATCH --gres=shard:10000            # GPU shards (~1 full L40S, max 12000)
#SBATCH -o logs/%x_%u_%j.out         # Output file
#SBATCH -e logs/%x_%u_%j.err         # Error file
#SBATCH -J qwen3vl_embed             # Job name

mkdir -p logs

. /home/youruser/miniconda3/etc/profile.d/conda.sh
conda activate tfg2

echo "=== STARTING FULL RUN: Qwen3-VL-Embedding-8B ==="
python src/experiments/run_qwen3vl.py \
    --data-dir   data/FakeVV_testset \
    --model-dir  src/models/Qwen3-VL-Embedding-8B \
    --qwen-repo  src/models/qwen3vl_embedding_repo \
    --output-dir experiments/open_source/results/qwen3vl_8b \
    --max-frames 16 \
    --fps 1.0 \
    --quantize
echo "=== FINISHED FULL RUN ==="
