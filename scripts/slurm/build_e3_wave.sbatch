#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /home/youruser/mavis-TFG
#SBATCH -t 1-12:00                    # 36h generoso (~7954 scene clips, regen tras borrar filas corruptas)
#SBATCH -p pg1tfg12
#SBATCH --account=g1tfg12
#SBATCH --qos=q_pg1tfg12
#SBATCH --gres=shard:12000
#SBATCH --mem=16G
#SBATCH -o logs/%x_%u_%j.out
#SBATCH -e logs/%x_%u_%j.err
#SBATCH -J e3_wave

mkdir -p logs
export PYTHONUNBUFFERED=1

. /home/youruser/miniconda3/etc/profile.d/conda.sh
conda activate tfg-wave

echo "=== STARTING: E3 scene embeddings — WAVE-7B ==="

python scripts/preprocessing/build_e3_scene_embeddings.py \
    --model      wave \
    --data-dir   data/GroundLie360 \
    --model-dir  src/models/WAVE-7B \
    --wave-repo  src/models/wave_repo \
    --beats-path src/models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
    --max-frames 16 \
    --quantize

echo "=== FINISHED: E3 scene embeddings — WAVE-7B ==="
