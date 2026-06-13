# Shock Detection via Physics-Informed Machine Learning

Arquitectura de Deep Learning para detección automática de ondas de choque en datos CFD de ONERA.

## Descripción

Este proyecto implementa una arquitectura sofisticada basada en:

1. **Autoencoder**: Aprende representación latente de física de flujos
2. **Mixture of Experts (MoE)**: Identifica múltiples regímenes físicos
3. **Sensor Virtual**: Predice probabilidad, intensidad y localización de choques

## Dataset

```
data/
├── X_train.npy    (81,361,488 × 9)   - Features de entrenamiento
├── X_test.npy     (40,680,744 × 9)   - Features de test
├── Ytrain.npy     (81,361,488 × 4)   - Outputs de entrenamiento
├── Ytest.npy      (40,680,744 × 4)   - Outputs de test
└── dataset.csv                        - Metadatos de simulaciones
```

### Features

**Input (9 features)**:
- Mach
- Angle of Attack (AoA)
- Pressure (Pi)
- Coordinates: x, y, z
- Surface normals: nx, ny, nz

**Output (4 features)**:
- Cp (Pressure coefficient)
- Cfx, Cfy, Cfz (Friction coefficients)

## Instalación

```bash
# Crear ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Uso Rápido

```bash
# Activar ambiente
source .venv/bin/activate

# Ejecutar pipeline completo
python main_train.py
```

El script:
1. Carga datos (con sampling estratégico para manejar 81M puntos)
2. Calcula variables derivadas (gradientes, Mach local, etc.)
3. Entrena Autoencoder
4. Entrena Mixture of Experts
5. Entrena Sensor Virtual
6. Genera reportes y visualizaciones

## Configuración

Editar `config.py` para ajustar:

```python
# Datos
DATA_CONFIG['train_sample_fraction'] = 0.1  # Usar 10% de training data

# Modelo
TRAINING_CONFIG['num_epochs'] = 50
TRAINING_CONFIG['batch_size'] = 512
TRAINING_CONFIG['learning_rate'] = 1e-3

# Preprocesamiento
PREPROCESSING_CONFIG['compute_pressure_gradient'] = True
PREPROCESSING_CONFIG['compute_mach_local'] = True
PREPROCESSING_CONFIG['compute_cp_loss'] = True
```

## Arquitectura

### Autoencoder

```
Input (19)
  ↓
Linear(19 → 128) + BatchNorm + LeakyReLU
  ↓
Linear(128 → 64) + BatchNorm + LeakyReLU
  ↓
Linear(64 → 32) + BatchNorm + LeakyReLU
  ↓
Latent Space (32 dimensions)
  ↓
[Decoder: espejo del encoder]
  ↓
Output (19)
```

### Mixture of Experts

```
Latent Space (32)
  ↓
┌─ Expert 1 (Adherent Flow)
├─ Expert 2 (Transonic)
├─ Expert 3 (Shock)
└─ Expert 4 (Separated)
  ↓
Gating Network (física + latente)
  ↓
Weighted Mixture
```

### Sensor Virtual

```
Latent Space (32)
  ↓
  ├→ Shock Head → P(shock)
  ├→ Intensity Head → Intensity
  └→ Separation Head → P(separation)
```

## Variables Derivadas (Tier 1)

El modelo calcula automáticamente:

1. **M_local**: Número de Mach local isentrópico
2. **∇P**: Gradiente de presión (suavizado)
3. **Cp_loss**: Pérdida de presión de remanso (indicador de choque)
4. **Shock_indicator**: Indicador combinado de presencia de choque
5. **Cf_magnitude**: Magnitud de fricción
6. **q_dynamic**: Presión dinámica
7. **Pi_normalized**: Presión normalizada por Mach
8. **AoA_normalized**: Ángulo de ataque normalizado
9. **grad_Cf**: Gradiente de fricción
10. **L_factor**: Factor de compresibilidad Laitone

## Resultados

Después de ejecutar, se generan:

```
outputs/
├── models/
│   ├── autoencoder_best.pt
│   └── moe_best.pt
├── results/
│   ├── autoencoder_evaluation.txt
│   └── training.log
└── plots/
    ├── ae_losses.png
    ├── predictions_vs_truth.png
    ├── latent_space.png
    └── reconstruction_error.png
```

## Métricas

El modelo reporta:

- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coeficiente de determinación por feature

## Características Principales

✅ **Manejo eficiente de datos enormes**: Memory mapping + sampling estratégico
✅ **Variables derivadas físicamente significativas**: Gradientes, números adimensionales, indicadores de régimen
✅ **Arquitectura modular**: Fácil de extender y modificar
✅ **Logging completo**: Seguimiento detallado del entrenamiento
✅ **Visualizaciones**: Pérdidas, predicciones, espacio latente, errores

## Optimizaciones para Producción

Para mejorar resultados:

1. **Aumentar `train_sample_fraction`**: 0.1 → 0.5 (más datos)
2. **Ajustar hiperparámetros**: Learning rate, batch size, épocas
3. **Agregar regularización**: Weight decay, dropout
4. **Data augmentation**: Ruido controlado durante entrenamiento
5. **Fine-tuning**: Entrenar en subconjuntos específicos

## Próximas Mejoras

- [ ] Graph Neural Networks (PointNet++, GCN)
- [ ] Symbolic Regression (pySR)
- [ ] SINDy para dinámica latente
- [ ] Validación cruzada en geometrías nuevas
- [ ] Análisis de robustez ante ruido
- [ ] Detección de anomalías en latente

## Referencias

Métodos implementados:

- Autoencoder basado en MLP (variante robusta)
- Mixture of Experts con gating físico
- Variables derivadas de CFD teórico
- Loss functions ponderados por gradientes

## Licencia

Proyecto académico - Uso libre

## Autor

Marta Arnabat Martín
