#!/bin/bash
# USEFUL COMMANDS

# ============ SETUP ============

# 1. Verificar Python
python3 --version

# 2. Crear ambiente
python3 -m venv .venv
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar instalación
python3 test_imports.py

# ============ DATA PREPROCESSING ============

# 5. Calcular features derivados (IMPRESCINDIBLE antes de entrenar)
python3 preprocess_data.py

# Verificar que se crearon
ls -lh data/*.npy

# ============ TRAINING ============

# 6. Entrenar pipeline completo
python3 main_train.py

# 7. Con logging detallado en archivo
python3 main_train.py 2>&1 | tee outputs/training_$(date +%Y%m%d_%H%M%S).log

# ============ INFERENCE & ANALYSIS ============

# 8. Análisis rápido de resultados
python3 infer.py --analyze --samples 100

# 9. Análisis con más samples
python3 infer.py --analyze --samples 1000

# ============ SYSTEM INFO ============

# 10. Ver uso de memoria
ps aux | grep python3

# 11. Monitor en tiempo real (macOS)
top -n 1 | head -20

# 12. Espacio disponible
df -h .

# 13. Tamaño de archivos de datos
du -sh data/

# ============ GIT OPERATIONS ============

# 14. Commit cambios
git add -A
git commit -m "Entrenamiento completado con AE + MoE"
git push origin main

# 15. Ver status
git status
git log --oneline -n 10

# ============ CLEANUP ============

# 16. Limpiar caché de Python
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# 17. Limpiar archivos temporales
rm -rf .pytest_cache *.pyc

# 18. Ver tamaño de outputs
du -sh outputs/

# ============ TROUBLESHOOTING ============

# 19. Si hay error con imports, reinstalar
pip install --upgrade torch numpy scikit-learn

# 20. Test de GPU (si disponible)
python3 -c "import torch; print('GPU available:', torch.cuda.is_available())"

# 21. Ver versiones instaladas
pip list | grep -E "torch|numpy|scipy|scikit"

# 22. Forzar CPU (si GPU causa problemas)
export CUDA_VISIBLE_DEVICES=""
python3 main_train.py

# ============ ADVANCED ============

# 23. Entrenar solo con datos de prueba (1%)
python3 << 'EOF'
from config import DATA_CONFIG
DATA_CONFIG['train_sample_fraction'] = 0.01
import main_train
main_train.main()
EOF

# 24. Perfiling de memoria
python3 -m memory_profiler main_train.py

# 25. Timing del entrenamiento
time python3 main_train.py

# 26. Ejecutar con output en tiempo real
python3 main_train.py &
tail -f outputs/training.log

# ============ DOCUMENTATION ============

# 27. Ver guía rápida
cat QUICK_START.md

# 28. Ver arquitectura
cat ARCHITECTURE.md

# 29. Ver README completo
cat README.md

# 30. Ver esta lista
cat COMMANDS.sh
