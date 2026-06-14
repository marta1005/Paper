#!/usr/bin/env bash
# ============================================================
# launch_gpu.sh  —  Lanzar entrenamiento completo con GPU
#
# Uso:
#   bash launch_gpu.sh            # modo producción (GPU)
#   bash launch_gpu.sh --demo     # modo demo rápida (CPU/GPU, 5% datos)
# ============================================================

set -euo pipefail

# ---------- configuración ----------
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$PROJECT_DIR/.venv"
LOG="$PROJECT_DIR/outputs/training_$(date +%Y%m%d_%H%M%S).log"

# ---------- modo ----------
DEMO=false
for arg in "$@"; do
  [[ "$arg" == "--demo" ]] && DEMO=true
done

# ---------- activar entorno ----------
if [[ -f "$VENV/bin/activate" ]]; then
  source "$VENV/bin/activate"
  echo "[launch] venv activado: $VENV"
else
  echo "[launch] AVISO: no se encontró .venv en $PROJECT_DIR"
  echo "         Ejecuta: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
fi

# ---------- verificar GPU ----------
python3 - <<'PYCHECK'
import torch
if torch.cuda.is_available():
    print(f"[launch] GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"         VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("[launch] AVISO: CUDA no disponible, se usará CPU")
PYCHECK

# ---------- ajustes de producción (sobrescriben config.py en runtime) ----------
if [[ "$DEMO" == false ]]; then
  echo "[launch] Modo PRODUCCIÓN: 100% datos, 100 épocas, batch 16384"
  export PAPER_TRAIN_FRACTION=1.0
  export PAPER_EPOCHS=100
  export PAPER_BATCH_SIZE=16384
  export PAPER_NUM_WORKERS=8
else
  echo "[launch] Modo DEMO: 5% datos, 5 épocas, batch 16384"
  export PAPER_TRAIN_FRACTION=0.05
  export PAPER_EPOCHS=5
  export PAPER_BATCH_SIZE=256
  export PAPER_NUM_WORKERS=0
fi

# ---------- lanzar ----------
mkdir -p "$PROJECT_DIR/outputs"
echo "[launch] Log → $LOG"
echo "[launch] Iniciando pipeline..."

cd "$PROJECT_DIR"
python3 main_train.py 2>&1 | tee "$LOG"

echo ""
echo "[launch] ✓ Entrenamiento finalizado."
echo "[launch]   Modelos:    outputs/models/"
echo "[launch]   Resultados: outputs/results/"
echo "[launch]   Plots:      outputs/plots/"
echo "[launch]   Log:        $LOG"
