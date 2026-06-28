#!/bin/bash
# download_models.sh — Download all model weights for MAVIS open-source experiments
#
# Run from repo root: bash scripts/setup/download_models.sh
# Requires: huggingface-cli (pip install huggingface-hub), git, wget

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"

echo "=== MAVIS Model Download Script ==="
echo "Repo root: $REPO_ROOT"
echo "Models dir: $MODELS_DIR"
mkdir -p "$MODELS_DIR"

# ---------------------------------------------------------------------------
# 1. Qwen3-VL-Embedding-2B (~5 GB)
# ---------------------------------------------------------------------------
echo ""
echo "--- 1. Qwen3-VL-Embedding-2B ---"
echo "Source: https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B"
echo "Size:   ~5 GB (FP16)"

if [ -d "$MODELS_DIR/Qwen3-VL-Embedding-28" ] && [ -f "$MODELS_DIR/Qwen3-VL-Embedding-2B/config.json" ]; then
    echo "Already downloaded. Skipping."
else
    huggingface-cli download Qwen/Qwen3-VL-Embedding-8B \
        --local-dir "$MODELS_DIR/Qwen3-VL-Embedding-8B"
    echo "Done: Qwen3-VL-Embedding-2B"
fi

# ---------------------------------------------------------------------------
# 2. WAVE-7B (~18.8 GB)
# ---------------------------------------------------------------------------
echo ""
echo "--- 2. WAVE-7B ---"
echo "Source: https://huggingface.co/tsinghua-ee/WAVE-7B"
echo "Size:   ~18.8 GB (4 x pytorch_model-*.bin)"

if [ -d "$MODELS_DIR/WAVE-7B" ] && [ -f "$MODELS_DIR/WAVE-7B/config.json" ]; then
    echo "Already downloaded. Skipping."
else
    huggingface-cli download tsinghua-ee/WAVE-7B \
        --local-dir "$MODELS_DIR/WAVE-7B"
    echo "Done: WAVE-7B"
fi

# ---------------------------------------------------------------------------
# 3. BEATs audio checkpoint (~300 MB)
# ---------------------------------------------------------------------------
echo ""
echo "--- 3. BEATs_iter3_plus.pt ---"
echo "Source: OneDrive (see link below if wget fails)"
echo "Size:   ~300 MB"
echo "Manual URL: https://1drv.ms/u/s!AqeByhGUtINrgcpj8ujXH1YUtxooEg?e=E9Ncea"

mkdir -p "$MODELS_DIR/BEATs"
if [ -f "$MODELS_DIR/BEATs/BEATs_iter3_plus.pt" ]; then
    echo "Already downloaded. Skipping."
else
    echo ""
    echo "NOTE: BEATs is hosted on OneDrive. Direct wget may not work."
    echo "If the download below fails, download manually from the URL above"
    echo "and place the file at: $MODELS_DIR/BEATs/BEATs_iter3_plus.pt"
    echo ""
    # Try direct download (may work with a direct link)
    # wget -O "$MODELS_DIR/BEATs/BEATs_iter3_plus.pt" "DIRECT_LINK_HERE"
    echo "Skipping automatic BEATs download. Download manually and place at:"
    echo "  $MODELS_DIR/BEATs/BEATs_iter3_plus.pt"
fi

# ---------------------------------------------------------------------------
# 4. WAVE source code repository
# ---------------------------------------------------------------------------
echo ""
echo "--- 4. WAVE source code (TCL606/WAVE) ---"
echo "Source: https://github.com/TCL606/WAVE"
echo "Size:   ~50 MB"

if [ -d "$MODELS_DIR/wave_repo" ] && [ -f "$MODELS_DIR/wave_repo/README.md" ]; then
    echo "Already cloned. Pulling latest..."
    cd "$MODELS_DIR/wave_repo" && git pull && cd "$REPO_ROOT"
else
    git clone https://github.com/TCL606/WAVE.git "$MODELS_DIR/wave_repo"
    echo "Done: wave_repo"
fi

# ---------------------------------------------------------------------------
# 5. Qwen3-VL-Embedding source code (needed for Qwen3VLEmbedder class)
# ---------------------------------------------------------------------------
echo ""
echo "--- 5. Qwen3-VL-Embedding source code (QwenLM/Qwen3-VL-Embedding) ---"
echo "Source: https://github.com/QwenLM/Qwen3-VL-Embedding"
echo "Size:   ~10 MB"

if [ -d "$MODELS_DIR/qwen3vl_embedding_repo" ] && [ -f "$MODELS_DIR/qwen3vl_embedding_repo/README.md" ]; then
    echo "Already cloned. Pulling latest..."
    cd "$MODELS_DIR/qwen3vl_embedding_repo" && git pull && cd "$REPO_ROOT"
else
    git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git "$MODELS_DIR/qwen3vl_embedding_repo"
    echo "Done: qwen3vl_embedding_repo"
fi

# ---------------------------------------------------------------------------
echo ""
echo "=== Download summary ==="
echo ""
echo "Qwen3-VL-8B: $MODELS_DIR/Qwen3-VL-Embedding-8B"
ls "$MODELS_DIR/Qwen3-VL-Embedding-8B/config.json" 2>/dev/null && echo "  status: OK" || echo "  status: MISSING"

echo "WAVE-7B:     $MODELS_DIR/WAVE-7B"
ls "$MODELS_DIR/WAVE-7B/config.json" 2>/dev/null && echo "  status: OK" || echo "  status: MISSING"

echo "BEATs:       $MODELS_DIR/BEATs/BEATs_iter3_plus.pt"
ls "$MODELS_DIR/BEATs/BEATs_iter3_plus.pt" 2>/dev/null && echo "  status: OK" || echo "  status: MISSING (download manually)"

echo "wave_repo:   $MODELS_DIR/wave_repo"
ls "$MODELS_DIR/wave_repo/README.md" 2>/dev/null && echo "  status: OK" || echo "  status: MISSING"

echo "qwen3vl_repo: $MODELS_DIR/qwen3vl_embedding_repo"
ls "$MODELS_DIR/qwen3vl_embedding_repo/README.md" 2>/dev/null && echo "  status: OK" || echo "  status: MISSING"

