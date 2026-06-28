#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /home/youruser/mavis-TFG
#SBATCH -t 00:20:00
#SBATCH -p pg1tfg12
#SBATCH --account=g1tfg12
#SBATCH --qos=q_pg1tfg12
#SBATCH --gres=shard:12000
#SBATCH --mem=16G
#SBATCH -o logs/%x_%u_%j.out
#SBATCH -e logs/%x_%u_%j.err
#SBATCH -J wave_readout_smoke

mkdir -p logs
export PYTHONUNBUFFERED=1
. /home/youruser/miniconda3/etc/profile.d/conda.sh
conda activate tfg-wave

python scripts/diagnostics/wave_readout_smoke.py \
    --model-dir   src/models/WAVE-7B \
    --wave-repo   src/models/wave_repo \
    --beats-path  src/models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
    --data-dir    data/M3A \
    --index-json  experiments/M3A/m3a_index_20000.json \
    --n 6 --max-frames 16 --quantize
