#!/bin/bash
#SBATCH --job-name=wave_smoke
#SBATCH --output=logs/%x_%u_%j.out
#SBATCH --error=logs/%x_%u_%j.err
#SBATCH -N 1
#SBATCH -D /home/youruser/mavis-TFG
#SBATCH --time=00:45:00
#SBATCH -p pg1tfg12
#SBATCH --account=g1tfg12
#SBATCH --qos=q_pg1tfg12
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

mkdir -p logs

. /home/youruser/miniconda3/etc/profile.d/conda.sh
conda activate tfg-wave

echo "=== WAVE TOKEN SMOKE TEST START ==="
python scripts/diagnostics/wave_token_smoke.py \
    --data-dir   data/GroundLie360 \
    --model-dir  src/models/WAVE-7B \
    --wave-repo  src/models/wave_repo \
    --beats-path src/models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
    --index-json experiments/GroundLie360/groundlie_index_filtered.json \
    --n-videos   8 \
    --max-frames 8 \
    --max-frame-side 224
echo "=== WAVE TOKEN SMOKE TEST END ==="
