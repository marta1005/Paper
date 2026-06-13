# QUICK START GUIDE

## 1. Instalación

```bash
cd /Users/martaarnabatmartin/Desktop/Paper
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Preprocesar Datos (Una sola vez)

```bash
python3 preprocess_data.py
```

Esto calcula **10 variables derivadas** para cada punto:
- Mach local (isentrópico)
- Gradientes de presión
- Pérdida de presión de remanso
- Indicadores de choque
- Factores de compresibilidad
- Y más...

Genera:
- `data/X_train_derived.npy` (81.3M × 19 features)
- `data/X_test_derived.npy` (40.6M × 19 features)

**Tiempo estimado**: ~10-15 minutos en CPU

## 3. Entrenar Modelos

```bash
python3 main_train.py
```

Pipeline automático:
1. **Stage 1**: Carga datos (5% para demo rápida)
2. **Stage 2**: Preprocesamiento (variables derivadas)
3. **Stage 3**: Entrena Autoencoder (32-dim latent space)
4. **Stage 4**: Entrena Mixture of Experts (4 regímenes)
5. **Stage 5**: Evaluación completa
6. **Stage 6**: Genera visualizaciones

Salida:
- `outputs/models/autoencoder_best.pt` - Modelo entrenado
- `outputs/results/autoencoder_evaluation.txt` - Métricas
- `outputs/plots/` - Gráficas (loss, predicciones, latent space, errores)

**Tiempo estimado**: ~30-60 minutos en CPU (2-5 min con GPU)

## 4. Analizar Resultados

```bash
python3 infer.py --analyze --samples 100
```

Muestra:
- Estadísticas del espacio latente
- Análisis del error de reconstrucción
- Distribuciones

## Configuración para Cambiar

Editar `config.py`:

```python
# Aumentar datos (más resultados, más tiempo)
DATA_CONFIG['train_sample_fraction'] = 0.1  # 10% en lugar de 5%

# Entrenar más épocas
TRAINING_CONFIG['num_epochs'] = 100  # En lugar de 20

# Batch size (trade-off: memoria vs velocidad)
TRAINING_CONFIG['batch_size'] = 512

# Learning rate
TRAINING_CONFIG['learning_rate'] = 1e-3
```

## Estructura de Archivos

```
Paper/
├── config.py                      # Configuración centralizada
├── main_train.py                  # Script principal de entrenamiento
├── preprocess_data.py             # Cálculo de features derivados
├── infer.py                       # Inferencia y análisis
├── test_imports.py                # Test rápido
├── requirements.txt               # Dependencias
├── README.md                       # Documentación completa
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Carga datos (maneja 81M filas)
│   ├── preprocessing.py          # Cálculo de features derivados
│   ├── models.py                 # Autoencoder + MoE + Sensor
│   ├── training.py               # Loops de entrenamiento
│   └── evaluation.py             # Métricas y visualización
│
├── data/
│   ├── X_train.npy              # 81.3M × 9 features
│   ├── X_test.npy               # 40.6M × 9 features
│   ├── Ytrain.npy               # 81.3M × 4 outputs (Cp, Cfx, Cfy, Cfz)
│   ├── Ytest.npy                # 40.6M × 4 outputs
│   ├── dataset.csv              # Metadatos
│   ├── X_train_derived.npy      # ← Generado por preprocess_data.py (19 features)
│   └── X_test_derived.npy       # ← Generado por preprocess_data.py (19 features)
│
└── outputs/
    ├── models/
    │   └── autoencoder_best.pt   # Modelo entrenado
    ├── results/
    │   ├── autoencoder_evaluation.txt
    │   └── training.log
    └── plots/
        ├── ae_losses.png
        ├── predictions_vs_truth.png
        ├── latent_space.png
        └── reconstruction_error.png
```

## Resultados Esperados

Después de entrenar con 5% de datos (~4M puntos) durante 20 épocas:

**Autoencoder**:
- RMSE: ~0.01-0.05 (dependiendo de scaling)
- Latent dim: 32 (captura estructura de física)
- Tiempo: ~40 min en CPU

**Mixture of Experts**:
- 4 regímenes identificados: Adherido, Transónico, Choque, Separado
- Gating network: asigna puntos a expertos
- Tiempo: ~15 min en CPU

**Visualizaciones generadas**:
- Curvas de pérdida (train/val)
- Scatter plot: predicciones vs valores reales
- Proyección 2D del espacio latente (PCA)
- Histograma de errores de reconstrucción

## Métricas Reportadas

```
MSE:   Mean Squared Error
RMSE:  Root Mean Squared Error
MAE:   Mean Absolute Error
R²:    Coeficiente de determinación por feature
       (R² = 1 es perfecto, R² = 0 es baseline)
```

## Optimizaciones para Mejores Resultados

1. **Aumentar datos**: `train_sample_fraction = 0.5`
2. **Más épocas**: `num_epochs = 100`
3. **Ajustar learning rate**: Probar `1e-4` a `1e-2`
4. **Early stopping**: Reduce épocas automáticamente si no mejora
5. **Fine-tuning**: Entrenar solo heads especializados después

## Troubleshooting

**Error: "File not found"**
```bash
# Verifica que estés en la carpeta correcta
cd /Users/martaarnabatmartin/Desktop/Paper
pwd  # Deberías ver: /Users/martaarnabatmartin/Desktop/Paper
```

**Error: "Out of memory"**
```python
# Reducir batch size en config.py
TRAINING_CONFIG['batch_size'] = 128  # En lugar de 256
```

**Entrenamiento muy lento**
```python
# Reducir datos en demo
DATA_CONFIG['train_sample_fraction'] = 0.01  # 1% en lugar de 5%
```

## Próximos Pasos (Investigación)

- [ ] Graph Neural Networks (PointNet++ para estructura)
- [ ] Symbolic Regression (pySR) en espacio latente
- [ ] SINDy para dinámicas latentes
- [ ] Detección de anomalías mejorada
- [ ] Transferencia a nuevas geometrías
- [ ] Validación en datos reales de túnel de viento

## Contacto / Preguntas

Ver `README.md` para referencias completas y detalles técnicos.
