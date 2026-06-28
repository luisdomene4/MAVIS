#!/bin/bash
#SBATCH -n 4                          # Number of cores
#SBATCH -N 1                          # All cores on one node
#SBATCH -D /home/youruser/mavis-TFG   # Working directory
#SBATCH -t 0-04:00                    # Runtime: 4 hours
#SBATCH -p pg1tfg12                   # Partition
#SBATCH --account=g1tfg12             # Account
#SBATCH --qos=q_pg1tfg12             # QoS
#SBATCH --mem 10G                   # 32GB memory
#SBATCH --gres=shard:4000            # GPU shards (~1 full L40S, max 12000)
#SBATCH -o logs/%x_%u_%j.out         # Output file
#SBATCH -e logs/%x_%u_%j.err         # Error file
#SBATCH -J compare_ge2_vs_qwen3vl    # Job name

mkdir -p logs

. /home/youruser/miniconda3/etc/profile.d/conda.sh
conda activate tfg2

echo "=== STARTING COMPARSION ==="
python src/evaluation/compare_ge2_vs_qwen3vl.py \
        --ge2-db experiments/FakeVV_testset/google_embeddings2/results/test_set_fakevvGE2.db \
        --qwen2b-db experiments/FakeVV_testset/open_source/results/qwen3vl_2b/qwen3vl_cache.db \
        --qwen8b-db experiments/FakeVV_testset/open_source/results/qwen3vl_8b/qwen3vl_cache.db \
        --index experiments/FakeVV_testset/google_embeddings2/test_index.json \
        --output experiments/FakeVV_testset/results \
        --max-duration 120
        
echo "=== FINISHED FULL RUN ==="
