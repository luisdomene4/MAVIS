#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -D /home/youruser/mavis-TFG
#SBATCH -t 0-01:00                    # ~40 API calls at 3 RPM ≈ 14 min
#SBATCH -p pg1tfg12
#SBATCH --account=g1tfg12
#SBATCH --qos=q_pg1tfg12
#SBATCH --mem 4G                      # No GPU — solo llamadas API
#SBATCH -o logs/%x_%u_%j.out
#SBATCH -e logs/%x_%u_%j.err
#SBATCH -J e2_ge2

mkdir -p logs
export PYTHONUNBUFFERED=1

. /home/youruser/miniconda3/etc/profile.d/conda.sh
conda activate tfg

# 2ª API key (cuota RPD independiente del diario). Exportada en ~/.bashrc del cluster.
echo "=== STARTING: E2 windows — GE2 (DB separada + 2ª key) ==="

python scripts/preprocessing/build_e2_windows.py \
    --model      ge2 \
    --api-key    "$GEMINI_API_KEY_EXP" \
    --output-db  experiments/GroundLie360/google_embeddings2/results/groundlie_ge2_segments.db \
    --rpm-target 3 \
    --rpd-limit  950

echo "=== FINISHED: E2 windows — GE2 ==="