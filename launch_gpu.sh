#!/bin/bash
#$ -N sensor
#$ -pe py 20
#$ -o sensor.out
#$ -e sensor.err
#$ -l m7j
#$ -q m7gpus
#$ -cwd
#
# Uso:
#   qsub launch_gpu.sh            # producción (GPU, 100% datos, 200 épocas)
#   bash launch_gpu.sh --demo     # demo rápida (5% datos, 5 épocas)

source /home/FlightPhysicsValidation/flowsimTest/dev_env.sh

# ---------- args ----------
DEMO=false
for arg in "$@"; do
  [[ "$arg" == "--demo" ]] && DEMO=true
done

# ---------- configuración ----------
PROJECT_DIR="/home/c05279/TIFON/ECCOMAS_2026/Paper-main"
LOG="$PROJECT_DIR/outputs/training_$(date +%Y%m%d_%H%M%S).log"
SENSOR_PKL="$PROJECT_DIR/outputs/models/shock_sensor_symbolic_surrogate_base.pkl"

mkdir -p "$PROJECT_DIR/outputs"

# ---------- verificar GPU ----------
python3.10 - <<'PYCHECK'
import torch
if torch.cuda.is_available():
    print(f"[launch] GPU: {torch.cuda.get_device_name(0)}  |  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("[launch] AVISO: CUDA no disponible, se usará CPU")
PYCHECK

# ---------- configuración de entrenamiento ----------
if [[ "$DEMO" == false ]]; then
  echo "[launch] Modo PRODUCCIÓN: 100% datos, 200 épocas, batch 16384"
  export PAPER_TRAIN_FRACTION=1.0
  export PAPER_EPOCHS=200
  export PAPER_BATCH_SIZE=16384
  export PAPER_NUM_WORKERS=8
else
  echo "[launch] Modo DEMO: 5% datos, 5 épocas, batch 256"
  export PAPER_TRAIN_FRACTION=0.05
  export PAPER_EPOCHS=5
  export PAPER_BATCH_SIZE=256
  export PAPER_NUM_WORKERS=0
fi

echo "[launch] Log → $LOG"
cd "$PROJECT_DIR"

# ---------- pipeline ----------
echo "[launch] [1/4] Preprocesando datos (9 → 16 features)..."
python3.10 preprocess_data.py 2>&1 | tee -a "$LOG" || exit 1

echo "[launch] [2/4] Entrenando AeroSurrogate (LayerNorm+SiLU, shock-gated residual, neural gate)..."
python3.10 main_train.py --stages surrogate 2>&1 | tee -a "$LOG" || exit 1

echo "[launch] [3/4] Destilando ShockIndicator → fórmula simbólica (PySR, 200 iteraciones)..."
python3.10 symbolic_regression.py \
    --mode surrogate \
    --samples 200000 \
    --iterations 200 2>&1 | tee -a "$LOG" || exit 1

echo "[launch] [4/4] Reentrenando con gate simbólico congelado..."
python3.10 main_train.py --stages surrogate \
    --symbolic-gate "$SENSOR_PKL" 2>&1 | tee -a "$LOG" || exit 1

echo ""
echo "[launch] ✓ Pipeline finalizado."
echo "[launch]   Modelos:          outputs/models/surrogate_best.pt          (gate neural)"
echo "[launch]                     outputs/models/surrogate_symbolic_best.pt  (gate simbólico)"
echo "[launch]   Sensor simbólico: $SENSOR_PKL"
echo "[launch]   Resultados SR:    outputs/results/symbolic_regression_surrogate_base.txt"
echo "[launch]   Log:              $LOG"
