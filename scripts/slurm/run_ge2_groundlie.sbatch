#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -D /home/youruser/mavis-TFG
#SBATCH -t 0-23:00                    # 23h — script se para solo al alcanzar el RPD limit
#SBATCH -p pg1tfg12
#SBATCH --account=g1tfg12
#SBATCH --qos=q_pg1tfg12
#SBATCH --mem 4G                      # Sin GPU, solo llamadas API
#SBATCH -o logs/%x_%u_%j.out
#SBATCH -e logs/%x_%u_%j.err
#SBATCH -J ge2_groundlie

# No GPU solicitada — este job solo hace llamadas a la API de Google.
# Requerimientos: pip install google-generativeai python-dotenv
# API key: GEMINI_API_KEY en .env en la raíz del repo o variable de entorno.

mkdir -p logs

. /home/youruser/miniconda3/etc/profile.d/conda.sh
conda activate tfg

# Verificar que google-generativeai está instalado
python -c "from google import genai" 2>/dev/null || {
    echo "ERROR: google-generativeai no encontrado. Instalar con:"
    echo "  pip install google-generativeai python-dotenv"
    exit 1
}

echo "=== STARTING: GE2 embeddings on GroundLie360 ==="
echo "    Target: 3 RPM (gap 20s), 950 RPD por sesión"
echo "    El script para solo cuando alcanza el límite de RPD."
echo "    Resubmitir mañana para continuar. El cache SQLite es reanudable."
echo ""

# Fases en orden:
#   0  = metadata del índice (sin API, siempre rápido)
#   A  = title text embeddings en batch (~35 llamadas API)
#   B  = transcript embeddings desde Qwen DB (~35 llamadas API)
#   C  = video embeddings uno a uno (1 llamada por vídeo, ~1722 en total)
#
# La primera vez: lanzar con --phases 0,A,B,C
# Si se para en C y se relanza: --phases C (A y B ya estarán completos)

python src/experiments/run_groundlie360_ge2.py \
    --data-dir   data/GroundLie360 \
    --output-dir experiments/GroundLie360/google_embeddings2/results \
    --index-json experiments/GroundLie360/groundlie_index_filtered.json \
    --bbox-json  experiments/GroundLie360/bbox_index.json \
    --qwen-db    experiments/GroundLie360/open_source/results/qwen3vl_2b/qwen3vl_cache.db \
    --phases     0,A,B,C \
    --rpm-target 3 \
    --rpd-limit  950 \
    --batch-size 25

echo ""
echo "=== FINISHED: GE2 groundlie session ==="
echo "    Si el log muestra [STOP] RPD limit reached:"
echo "    Resubmitir con: sbatch scripts/slurm/run_ge2_groundlie.sbatch"
