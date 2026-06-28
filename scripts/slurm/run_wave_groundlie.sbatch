#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /home/youruser/mavis-TFG
#SBATCH -t 1-12:00                    # 36h generoso (regen video+audio+AV tras borrar filas corruptas)
#SBATCH -p pg1tfg12
#SBATCH --account=g1tfg12
#SBATCH --qos=q_pg1tfg12
#SBATCH --gres=shard:12000
#SBATCH --mem=16G
#SBATCH -o logs/%x_%u_%j.out
#SBATCH -e logs/%x_%u_%j.err
#SBATCH -J wave_groundlie

mkdir -p logs

. /home/youruser/miniconda3/etc/profile.d/conda.sh
conda activate tfg-wave

echo "=== STARTING: WAVE-7B on GroundLie360 ==="

python src/experiments/run_groundlie360_wave.py \
    --data-dir   data/GroundLie360 \
    --model-dir  src/models/WAVE-7B \
    --wave-repo  src/models/wave_repo \
    --beats-path src/models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
    --output-dir experiments/GroundLie360/open_source/results/WAVE7B \
    --index-json experiments/GroundLie360/groundlie_index_filtered.json \
    --bbox-json  experiments/GroundLie360/bbox_index.json \
    --qwen-db    experiments/GroundLie360/open_source/results/qwen3vl_2b/qwen3vl_cache.db \
    --max-frames 16 \
    --quantize

echo "=== FINISHED: WAVE-7B on GroundLie360 ==="
