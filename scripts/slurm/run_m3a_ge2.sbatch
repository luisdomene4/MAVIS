#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -D /home/youruser/mavis-TFG
#SBATCH -t 0-23:00                    # script se para solo al alcanzar el RPD limit
#SBATCH -p pg1tfg12
#SBATCH --account=g1tfg12
#SBATCH --qos=q_pg1tfg12
#SBATCH --mem 4G                      # Sin GPU, solo API
#SBATCH -o logs/%x_%u_%j.out
#SBATCH -e logs/%x_%u_%j.err
#SBATCH -J ge2_m3a

mkdir -p logs

. /home/youruser/miniconda3/etc/profile.d/conda.sh
conda activate tfg

python -c "from google import genai" 2>/dev/null || {
    echo "ERROR: google-generativeai no encontrado. pip install google-generativeai python-dotenv"
    exit 1
}

# ─────────── ROTACIÓN DE API KEY ───────────
# Por defecto la 2ª key (GEMINI_API_KEY_EXP …Iv_w). Para usar la otra, sin editar:
#   KEY_VAR=GEMINI_API_KEY sbatch scripts/slurm/run_m3a_ge2.sbatch
KEY_VAR="${KEY_VAR:-GEMINI_API_KEY_EXP}"
# ────────────────────────────────────────────
set -a; [ -f .env ] && . ./.env; set +a
export GEMINI_API_KEY="${!KEY_VAR}"
echo "GE2 usando $KEY_VAR (…${GEMINI_API_KEY: -4})"

INDEX="${M3A_INDEX:-experiments/M3A/m3a_index_2500.json}"
echo "=== STARTING: GE2 embeddings on M3A ($INDEX) ==="
echo "    El script para solo al alcanzar el RPD limit. Resubmitir mañana (cache reanudable)."

# Fases: 0=metadata  A=summary  N=NEM(+MTG)  B=transcript(desde Qwen DB)  C=video(1 call/vid)
# La quota de Google da ~1000 embeds/día. Estrategia: completitud por vídeo
# (summary+transcript+vídeo) sobre un subset representativo, NEM al final.
# Overrides por variable:
#   GE2_PHASES=0,A,B,C  → summary+transcript+vídeo (sin NEM)
#   GE2_LIMIT=1000      → solo los primeros 1000 del índice ordenado (representativo)
PHASES="${GE2_PHASES:-0,A,N,B,C}"
LIMIT_ARG=""
[ -n "${GE2_LIMIT:-}" ] && LIMIT_ARG="--limit ${GE2_LIMIT}"
python src/experiments/run_m3a_ge2.py \
    --data-dir   data/M3A \
    --output-dir experiments/M3A/google_embeddings2/results \
    --index-json "$INDEX" \
    --qwen-db    experiments/M3A/open_source/results/qwen3vl_2b/qwen3vl_cache.db \
    --phases     "$PHASES" \
    $LIMIT_ARG \
    --rpm-target 3 \
    --rpd-limit  950 \
    --batch-size 25

echo "=== FINISHED: GE2 M3A session ==="
echo "    Si el log muestra [STOP] RPD limit reached: sbatch scripts/slurm/run_m3a_ge2.sbatch"
