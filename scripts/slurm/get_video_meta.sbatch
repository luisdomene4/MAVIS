#!/bin/bash
#SBATCH -n 2                          # 2 cores (ffprobe is sequential, 1 extra for OS)
#SBATCH -N 1                          # Single node
#SBATCH -D /home/youruser/mavis-TFG   # Working directory
#SBATCH -t 0-01:00                    # 1 hour (ffprobe on 2044 videos takes ~10-15 min)
#SBATCH -p pg1tfg12                   # Partition
#SBATCH --account=g1tfg12             # Account
#SBATCH --qos=q_pg1tfg12             # QoS
#SBATCH --mem 2G                      # Minimal memory (no GPU needed)
#SBATCH -o logs/%x_%u_%j.out         # Output file
#SBATCH -e logs/%x_%u_%j.err         # Error file
#SBATCH -J get_video_meta             # Job name

mkdir -p logs

. /home/youruser/miniconda3/etc/profile.d/conda.sh
conda activate tfg

echo "=== get_video_meta: GroundLie360 ==="
python scripts/preprocessing/get_video_meta.py
echo "=== DONE ==="
