#!/bin/bash
# create_env_wave.sh — Create isolated conda environment for WAVE-7B experiments
#
# Run on cluster: bash scripts/setup/create_env_wave.sh
# Time: ~5 min (mostly torch download)
#
# This keeps tfg2 untouched (Qwen3VL) while giving WAVE its required
# transformers>=4.51.3 + torch 2.6.

set -e

ENV_NAME="tfg-wave"

echo "=== Creating conda env: $ENV_NAME ==="

# Create env with Python 3.10 (same as tfg2 for consistency)
conda create -n "$ENV_NAME" python=3.10 -y

# Activate
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Python: $(python --version)"
echo "Env: $CONDA_DEFAULT_ENV"

# Core: PyTorch with CUDA (matching cluster L40S drivers)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# WAVE requirements
pip install transformers==4.51.3
pip install accelerate==1.7.0
pip install liger_kernel==0.5.10

# Our pipeline deps
pip install av              # PyAV: audio extraction from video
pip install numpy pandas scikit-learn matplotlib

# BitsAndBytesConfig for potential INT4 quantization
pip install bitsandbytes

echo ""
echo "=== Environment $ENV_NAME ready ==="
echo ""
echo "Verify:"
echo "  conda activate $ENV_NAME"
echo "  python -c \"from transformers.image_utils import VideoInput; print('OK')\""
echo ""
echo "Then submit:"
echo "  sbatch scripts/slurm/run_wave.sbatch"
