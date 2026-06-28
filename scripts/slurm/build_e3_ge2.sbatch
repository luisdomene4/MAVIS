#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -D /home/youruser/mavis-TFG
#SBATCH -t 0-12:00                    # subset: ~620 scene clips at 3 RPM ≈ 3.5h
#SBATCH -p pg1tfg12
#SBATCH --account=g1tfg12
#SBATCH --qos=q_pg1tfg12
#SBATCH --mem 4G                      # No GPU — ffmpeg cut + API calls
#SBATCH -o logs/%x_%u_%j.out
#SBATCH -e logs/%x_%u_%j.err
#SBATCH -J e3_ge2

mkdir -p logs
export PYTHONUNBUFFERED=1

. /home/youruser/miniconda3/etc/profile.d/conda.sh
conda activate tfg

# 2ª API key (cuota RPD independiente del diario). Exportada en ~/.bashrc del cluster.
echo "=== STARTING: E3 scene embeddings — GE2 (subset only, DB separada + 2ª key) ==="

python scripts/preprocessing/build_e3_scene_embeddings.py \
    --model      ge2 \
    --api-key    "$GEMINI_API_KEY_EXP" \
    --output-db  experiments/GroundLie360/google_embeddings2/results/groundlie_ge2_segments.db \
    --data-dir   data/GroundLie360 \
    --subset-json experiments/GroundLie360/e3_ge2_subset.json \
    --rpm-target 3 \
    --rpd-limit  950

echo "=== FINISHED: E3 scene embeddings — GE2 ==="
echo "    Si [STOP] RPD limit: resubmit con sbatch scripts/slurm/build_e3_ge2.sbatch"
