#!/bin/bash
#SBATCH -n 4                          # Number of cores
#SBATCH -N 1                          # All cores on one node
#SBATCH -D /home/youruser/mavis-TFG   # Working directory
#SBATCH -t 1-00:00                    # 24h generoso (regen video+audio+AV de FakeVV tras borrar filas corruptas)
#SBATCH -p pg1tfg12                   # Partition
#SBATCH --account=g1tfg12             # Account
#SBATCH --qos=q_pg1tfg12             # QoS
#SBATCH --gres=shard:12000
#SBATCH --mem=16G
#SBATCH -o logs/%x_%u_%j.out         # Output file
#SBATCH -e logs/%x_%u_%j.err         # Error file
#SBATCH -J wave_embed                 # Job name

mkdir -p logs

. /home/youruser/miniconda3/etc/profile.d/conda.sh
conda activate tfg-wave

echo "=== STARTING FULL RUN: WAVE-7B ==="
python src/experiments/run_wave.py \
    --data-dir   data/FakeVV_testset \
    --model-dir  src/models/WAVE-7B \
    --wave-repo  src/models/wave_repo \
    --beats-path src/models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
    --output-dir experiments/open_source/results/WAVE7B \
    --max-frames 16 \
    --quantize
echo "=== FINISHED FULL RUN ==="
