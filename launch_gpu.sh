# ============================================================
# launch_gpu.sh  —  Lanzar entrenamiento completo con GPU
#
# Uso:
#   bash launch_gpu.sh            # modo producción (GPU)
#   bash launch_gpu.sh --demo     # modo demo rápida (CPU/GPU, 5% datos)
# ============================================================

#!/bin/bash
#$ -N sensor # Name of the job
#$ -pe py 20 # Number of threads
#$ -o sensor.out # Log for stdout
#$ -e sensor.err # Log for stderr
#$ -l m7j # clave siempre para ir a los nodos de mcn32*
#$ -q m7gpus # si quiero ir a esa cola (2 nodos de gpus mcn320 / mcn321)
#$ -cwd

source /home/FlightPhysicsValidation/flowsimTest/dev_env.sh

# ---------- configuración ----------
PROJECT_DIR="/home/c05279/TIFON/ECCOMAS_2026/Paper-main"
LOG="$PROJECT_DIR/outputs/training_$(date +%Y%m%d_%H%M%S).log"

# ---------- verificar GPU ----------
python3.10 - <<'PYCHECK'
import torch
if torch.cuda.is_available():
    print(f"[launch] GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"         VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("[launch] AVISO: CUDA no disponible, se usará CPU")
PYCHECK

DEMO=false
# ---------- ajustes de producción (sobrescriben config.py en runtime) ----------
if [[ "$DEMO" == false ]]; then
  echo "[launch] Modo PRODUCCIÓN: 100% datos, 200 épocas, batch 16384"
  export PAPER_TRAIN_FRACTION=1.0
  export PAPER_EPOCHS=200
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

cd "$PROJECT_DIR"

echo "[launch] [1/4] Preprocesando datos (9 -> 16 features)..."
python3.10 preprocess_data.py 2>&1 | tee -a "$LOG"

echo "[launch] [2/4] Entrenando Surrogate (ShockIndicator + MoE, neural gate)..."
python3.10 main_train.py --stages surrogate 2>&1 | tee -a "$LOG"

echo "[launch] [3/4] Sensor simbólico con K-NN gradient labels (Decision Tree)..."
python3.10 symbolic_regression.py \
    --mode physics --fallback \
    --knn-labels --knn-k 50 --knn-percentile 85 \
    --samples 200000 2>&1 | tee -a "$LOG"

SENSOR_PKL="$PROJECT_DIR/outputs/models/shock_sensor_symbolic_physics_knn.pkl"
echo "[launch] [4/4] Reentrenando MoE con gate simbólico (ShockIndicator congelado)..."
python3.10 main_train.py --stages surrogate \
    --symbolic-gate "$SENSOR_PKL" 2>&1 | tee -a "$LOG"

echo ""
echo "[launch] ✓ Pipeline finalizado."
echo "[launch]   Modelos:              outputs/models/surrogate_best.pt (neural gate)"
echo "[launch]                         outputs/models/surrogate_symbolic_best.pt (symbolic gate)"
echo "[launch]   Sensor simbólico:     $SENSOR_PKL"
echo "[launch]   Resultados SR:        outputs/results/symbolic_regression_*.txt"
echo "[launch]   Plots:                outputs/plots/"
echo "[launch]   Log:                  $LOG"
