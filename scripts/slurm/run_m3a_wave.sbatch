#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /home/youruser/mavis-TFG
#SBATCH -t 2-00:00                    # 48h generoso (regen video sobre 20k tras borrar filas corruptas)
#SBATCH -p pg1tfg12
#SBATCH --account=g1tfg12
#SBATCH --qos=q_pg1tfg12
#SBATCH --gres=shard:12000
#SBATCH --mem=16G
#SBATCH -o logs/%x_%u_%j.out
#SBATCH -e logs/%x_%u_%j.err
#SBATCH -J wave_m3a

mkdir -p logs
export PYTHONUNBUFFERED=1

. /home/youruser/miniconda3/etc/profile.d/conda.sh
conda activate tfg-wave

INDEX="${M3A_INDEX:-experiments/M3A/m3a_index_20000.json}"   # default 20k (WAVE M3A regen); evita --export que daba 'user env retrieval failed'
echo "=== STARTING: WAVE-7B on M3A ($INDEX) ==="

# NOTE: no audio/AV phases — audio is represented via transcript (read from Qwen DB).
# Run run_m3a_qwen3vl_2b first so the transcript modality can be populated.
python src/experiments/run_m3a_wave.py \
    --data-dir   data/M3A \
    --model-dir  src/models/WAVE-7B \
    --wave-repo  src/models/wave_repo \
    --beats-path src/models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
    --output-dir experiments/M3A/open_source/results/WAVE7B \
    --index-json "$INDEX" \
    --qwen-db    experiments/M3A/open_source/results/qwen3vl_2b/qwen3vl_cache.db \
    --max-frames 16 \
    --quantize

echo "=== FINISHED: WAVE-7B on M3A ==="
