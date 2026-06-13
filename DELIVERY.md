# 📋 DOCUMENTO DE ENTREGA

**Proyecto**: Shock Detection via Physics-Informed Machine Learning  
**Fecha**: 13 de junio de 2026  
**Estado**: ✅ Código listo para ejecutar (entrenamiento en progreso)

---

## 🎯 QUÉ SE HA ENTREGADO

Una arquitectura completa de Deep Learning + Physics-Informed ML para detectar ondas de choque en datos CFD de ONERA.

### Componentes Principales

```
✅ Autoencoder (32-dim latent space)
✅ Mixture of Experts (4 regímenes: adherido, transónico, choque, separado)
✅ Sensor Virtual (predicción de choque, intensidad, separación)
✅ Features derivados (10 variables físicamente significativas)
✅ Data loader optimizado (maneja 81M de puntos)
✅ Training pipeline completo con early stopping
✅ Evaluación con visualizaciones
```

---

## 📁 ESTRUCTURA DEL PROYECTO

```
Paper/
├── 📄 QUICK_START.md          ← LEE ESTO PRIMERO
├── 📄 ARCHITECTURE.md         ← Detalles técnicos
├── 📄 README.md               ← Documentación completa
├── 📄 COMMANDS.sh             ← Comandos útiles
│
├── ⚙️ Configuración
│   └── config.py              ← Todos los parámetros (EDITAR AQUÍ)
│
├── 🚀 Scripts principales
│   ├── main_train.py          ← EJECUTAR ESTO para entrenar
│   ├── preprocess_data.py     ← Calcular features derivados (YA EJECUTADO ✅)
│   ├── infer.py               ← Inferencia y análisis
│   └── test_imports.py        ← Test rápido
│
├── 📦 Código modular (src/)
│   ├── data_loader.py         ← Carga datos 81M puntos eficientemente
│   ├── preprocessing.py       ← Cálculo de variables derivadas
│   ├── models.py              ← Arquitectura (AE + MoE + Sensor)
│   ├── training.py            ← Loops de entrenamiento
│   └── evaluation.py          ← Métricas y visualización
│
├── 📊 Data
│   ├── X_train.npy            (81.3M × 9 features originales)
│   ├── X_test.npy             (40.6M × 9 features originales)
│   ├── Ytrain.npy             (81.3M × 4 outputs: Cp, Cfx, Cfy, Cfz)
│   ├── Ytest.npy              (40.6M × 4 outputs)
│   ├── dataset.csv            (metadatos)
│   ├── X_train_derived.npy    ✅ (81.3M × 19 features derivados) ← GENERADO
│   └── X_test_derived.npy     ✅ (40.6M × 19 features derivados) ← GENERADO
│
├── 📈 Outputs (generados después de entrenar)
│   ├── models/
│   │   └── autoencoder_best.pt
│   ├── results/
│   │   ├── autoencoder_evaluation.txt
│   │   └── training.log
│   └── plots/
│       ├── ae_losses.png
│       ├── predictions_vs_truth.png
│       ├── latent_space.png
│       └── reconstruction_error.png
│
└── requirements.txt           ← Dependencias
```

---

## ⚡ INSTRUCCIONES RÁPIDAS

### 1️⃣ Primera vez: Setup (5 minutos)

```bash
cd /Users/martaarnabatmartin/Desktop/Paper
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 test_imports.py  # Verificar que funciona
```

### 2️⃣ Preprocesar datos (⚠️ IMPORTANTE - YA HECHO ✅)

```bash
python3 preprocess_data.py  # Toma ~10-15 min
# Genera: X_train_derived.npy, X_test_derived.npy
```

**Estado**: ✅ **YA COMPLETADO**  
Archivos generados:
- `data/X_train_derived.npy` (81.3M × 19)
- `data/X_test_derived.npy` (40.6M × 19)

### 3️⃣ Entrenar modelo (HACERLO AHORA)

```bash
python3 main_train.py
```

**Proceso**:
- Stage 1: Carga datos (~30 seg)
- Stage 2: Preprocesamiento (ya hecho)
- Stage 3: Autoencoder training (10-30 min en CPU)
- Stage 4: MoE training (5-10 min)
- Stage 5: Evaluación (2-5 min)
- Stage 6: Visualizaciones (1 min)

**Salida esperada**: Archivos en `outputs/`

### 4️⃣ Analizar resultados

```bash
python3 infer.py --analyze --samples 100
# Muestra estadísticas del espacio latente
```

---

## 🔧 CONFIGURACIÓN RECOMENDADA PARA PAPER

Si necesitas **mejores resultados para el paper**:

Editar `config.py`:

```python
# Línea ~19: Aumentar datos
'train_sample_fraction': 0.2,  # Cambiar de 0.05 a 0.2 (20% = 16M puntos)

# Línea ~28: Más épocas
'num_epochs': 50,  # Cambiar de 20 a 50

# Línea ~27: Batch size mayor (si hay memoria)
'batch_size': 512,  # Cambiar de 256 a 512
```

Luego ejecutar:
```bash
python3 main_train.py
```

**Tiempo estimado**: 1-2 horas en CPU, 10-15 min en GPU

---

## 📊 VARIABLES DERIVADAS CALCULADAS

El código calcula automáticamente 10 variables físicamente significativas:

| # | Variable | Significado Físico |
|---|----------|-------------------|
| 1 | M_local | Número de Mach local (isentrópico) |
| 2 | grad_P | Gradiente de presión (indica transiciones) |
| 3 | Cp_loss | Pérdida de presión de remanso (choque) |
| 4 | shock_indicator | Indicador combinado de choque |
| 5 | Cf_magnitude | Magnitud de fricción (separación) |
| 6 | q_dynamic | Presión dinámica |
| 7 | Pi_normalized | Presión normalizada |
| 8 | AoA_normalized | Ángulo normalizado |
| 9 | grad_Cf | Gradiente de fricción |
| 10 | L_factor | Factor Laitone (compresibilidad) |

**Input original**: 9 features  
**Input después de preprocessing**: 19 features (9 + 10 derivados)

---

## 🏗️ ARQUITECTURA RESUMIDA

```
Input (19 features)
      ↓
  [AUTOENCODER]
  ├─ Encoder: 19 → 128 → 64 → 32
  ├─ Latent: 32 dimensions
  └─ Decoder: 32 → 64 → 128 → 19
      ↓
  Latent Space (32 dims)
      ↓
  [MIXTURE OF EXPERTS]
  ├─ 4 Expertos especializados
  ├─ Gating network (física-informado)
  └─ Salida: predicciones especializadas
      ↓
  [SENSOR VIRTUAL]
  ├─ Shock Probability
  ├─ Shock Intensity
  └─ Separation Probability
```

**Parámetros totales**: ~114,000 (AE 28K + MoE 86K)

---

## 📈 MÉTRICAS ESPERADAS

Con 5% de datos (demo) durante 20 épocas:

```
Autoencoder:
  RMSE: 0.01 - 0.05
  MAE:  0.005 - 0.03
  R²:   0.70 - 0.95 (depende del feature)

MoE:
  Gating accuracy: 70-85%
  Régimen identification: 60-75%
```

**Nota**: Mejoran significativamente con más datos y épocas

---

## 🚀 PRÓXIMOS PASOS PARA PAPER

### Corto plazo (resultados inmediatos):
1. ✅ Ejecutar `main_train.py`
2. ✅ Analizar resultados con `infer.py`
3. ✅ Generar figuras de `outputs/plots/`
4. ✅ Documentar métricas

### Mediano plazo (mejoras):
1. Aumentar `train_sample_fraction` a 0.5
2. Entrenar más épocas
3. Fine-tuning del sensor virtual
4. Análisis de espacio latente

### Largo plazo (investigación):
1. Implementar GNN (Graph Neural Networks)
2. Aplicar pySR (Symbolic Regression)
3. Transferencia a nuevas geometrías
4. Validación experimental

---

## ⚠️ IMPORTANTE: PROBLEMAS CONOCIDOS

### 1. Entrenamiento lento (CPU)
**Solución**: Usar GPU
```bash
# En Mac con GPU Metal Performance Shaders:
# Automático si disponible
# Si no: usar GPU externa
```

### 2. Out of memory
**Solución**: Reducir batch size en config.py
```python
TRAINING_CONFIG['batch_size'] = 128  # en lugar de 256
```

### 3. Features derivados no se calculan
**Solución**: Ejecutar primero `preprocess_data.py`
```bash
python3 preprocess_data.py  # Ya hecho ✅
```

---

## 📚 DOCUMENTACIÓN

- **QUICK_START.md**: Guía paso a paso (RECOMENDADO)
- **ARCHITECTURE.md**: Detalles técnicos de la arquitectura
- **README.md**: Documentación completa
- **COMMANDS.sh**: Comandos útiles

---

## 🔍 VERIFICACIÓN DE EJECUCIÓN

Para verificar que todo funciona:

```bash
# 1. Test rápido de imports
python3 test_imports.py

# 2. Chequear que se crearon archivos derivados
ls -lh data/X_*_derived.npy

# 3. Ver logs de entrenamiento en tiempo real
tail -f outputs/training.log  # (después de iniciar main_train.py)

# 4. Monitor de memoria/CPU
top -n 1 | head -20
```

---

## 💾 ARCHIVOS GENERADOS DESPUÉS DE ENTRENAR

```
outputs/
├── models/
│   └── autoencoder_best.pt        (~30KB)
├── results/
│   ├── autoencoder_evaluation.txt  (métricas)
│   └── training.log               (log completo)
└── plots/
    ├── ae_losses.png              (curvas de pérdida)
    ├── predictions_vs_truth.png    (comparación)
    ├── latent_space.png           (PCA 2D)
    └── reconstruction_error.png   (distribución de errores)
```

---

## 🎓 PARA INCLUIR EN PAPER

### Sección Metodología:
- Describir arquitectura AE + MoE + Sensor
- Explicar variables derivadas
- Mostrar diagrama del pipeline

### Sección Resultados:
- Métricas (RMSE, MAE, R²)
- Visualizaciones de `outputs/plots/`
- Análisis del espacio latente
- Performance en regímenes diferentes

### Sección Conclusiones:
- Capacidad de generalización
- Limitaciones observadas
- Trabajo futuro (GNN, pySR, etc.)

---

## ✅ CHECKLIST DE EJECUCIÓN

- [ ] Setup ambiente (`.venv`, pip install)
- [ ] Verificar imports (`test_imports.py`)
- [ ] Verificar datos derivados existen
- [ ] Ejecutar `main_train.py`
- [ ] Esperar a que termine (1-2 horas CPU)
- [ ] Analizar resultados (`infer.py --analyze`)
- [ ] Recopilar figuras de `outputs/plots/`
- [ ] Documentar métricas
- [ ] Escribir paper con resultados

---

## 📞 SOPORTE RÁPIDO

**Problema**: Comando no encontrado  
**Solución**: `source .venv/bin/activate`

**Problema**: No hay datos derivados  
**Solución**: `python3 preprocess_data.py`

**Problema**: Out of memory  
**Solución**: Reducir `batch_size` o `train_sample_fraction`

**Problema**: Entrenamiento muy lento  
**Solución**: Usar GPU o esperar (CPU es lento pero funciona)

---

## 🎯 RESUMEN

✅ **Código listo para ejecutar**  
✅ **Arquitectura completa (AE + MoE + Sensor)**  
✅ **Features derivados calculados**  
✅ **Data loaders optimizados para 81M puntos**  
✅ **Documentación exhaustiva**  

**Próximo paso**: Ejecuta `python3 main_train.py` y espera resultados.

---

Marta Arnabat Martín | 13 de junio de 2026
