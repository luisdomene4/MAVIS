#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /home/youruser/mavis-TFG
#SBATCH -t 0-01:00                    # 1h max for smoke test
#SBATCH -p pg1tfg12
#SBATCH --account=g1tfg12
#SBATCH --qos=q_pg1tfg12
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o logs/%x_%u_%j.out
#SBATCH -e logs/%x_%u_%j.err
#SBATCH -J wave_smoke_res

mkdir -p logs

. /home/youruser/miniconda3/etc/profile.d/conda.sh
conda activate tfg-wave

echo "=== WAVE RESOLUTION SMOKE TEST ==="
echo "Time: $(date)"
echo "Host: $(hostname)"

# Force Hugging Face offline mode to prevent connection timeouts on cluster nodes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Force single GPU — with gres=shard SLURM may not set CUDA_VISIBLE_DEVICES
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "CUDA_VISIBLE_DEVICES not set by SLURM, forcing to GPU 0"
else
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (set by SLURM)"
fi

nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || \
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

python -u scripts/preprocessing/smoke_wave_resolution.py \
    --data-dir   data/FakeVV_testset \
    --model-dir  src/models/WAVE-7B \
    --wave-repo  src/models/wave_repo \
    --beats-path src/models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
    --limit 5

echo ""
echo "=== FINISHED ==="
echo "Time: $(date)"
