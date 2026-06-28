#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /home/youruser/mavis-TFG
#SBATCH -t 0-01:00                    # text-only, ~40 embeds; model load dominates
#SBATCH -p pg1tfg12
#SBATCH --account=g1tfg12
#SBATCH --qos=q_pg1tfg12
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o logs/%x_%u_%j.out
#SBATCH -e logs/%x_%u_%j.err
#SBATCH -J e2_wave

mkdir -p logs
export PYTHONUNBUFFERED=1

. /home/youruser/miniconda3/etc/profile.d/conda.sh
conda activate tfg-wave

echo "=== STARTING: E2 windows — WAVE-7B ==="

python scripts/preprocessing/build_e2_windows.py \
    --model      wave \
    --model-dir  src/models/WAVE-7B \
    --wave-repo  src/models/wave_repo \
    --beats-path src/models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
    --quantize

echo "=== FINISHED: E2 windows — WAVE-7B ==="