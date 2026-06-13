# REFERENCE: Arquitectura del Sistema

## Visión General

```
          X_train (81.3M × 9 features)
                ↓
        [PREPROCESSING]
        Calcula 10 features derivados
                ↓
       X_train (81.3M × 19 features)
                ↓
        ┌─────────────────────────────┐
        │   AUTOENCODER (32-dim)      │
        │  Input: 19 features         │
        │  Latent: 32 dimensions      │
        │  Output: 19 features (recon)│
        └─────────────────────────────┘
                ↓
        Espacio latente aprendido
        (captura "física" del flujo)
                ↓
        ┌──────────────────────────────┐
        │  MIXTURE OF EXPERTS (MoE)    │
        │                              │
        │  Gating network:             │
        │  - Recibe: z (latente) +     │
        │    indicadores físicos       │
        │  - Asigna a expertos         │
        │                              │
        │  Expertos (4):               │
        │  1. Flujo adherido           │
        │  2. Transónico suave         │
        │  3. Con choque               │
        │  4. Separado                 │
        └──────────────────────────────┘
                ↓
        Predicciones especializadas
                ↓
        ┌──────────────────────────────────────┐
        │  SENSOR VIRTUAL                      │
        │  ├─ Shock Head: P(shock)             │
        │  ├─ Intensity Head: Intensidad       │
        │  └─ Separation Head: P(separation)   │
        └──────────────────────────────────────┘
                ↓
        Salida final:
        - Probabilidad de choque
        - Intensidad estimada
        - Probabilidad de separación
```

## 1. AUTOENCODER

### Arquitectura

```
ENCODER:
Input (19)
  → Linear(19→128) + BatchNorm + LeakyReLU
  → Linear(128→64) + BatchNorm + LeakyReLU
  → Linear(64→32) + BatchNorm + LeakyReLU
  → Latent (32)

DECODER (espejo del encoder):
Latent (32)
  → Linear(32→64) + BatchNorm + LeakyReLU
  → Linear(64→128) + BatchNorm + LeakyReLU
  → Linear(128→19)
  → Output (19)
```

### Parámetros

- **Total parámetros**: ~28,800
- **Batch size**: 256
- **Loss**: MSE ponderado por gradiente de presión
- **Optimizer**: Adam(lr=1e-3, weight_decay=1e-5)

### Propósito

Comprimir datos de alta dimensión (19) en representación latente (32) que capture la esencia física del flujo.

**Beneficios**:
1. Reducción de dimensionalidad
2. Denoising implícito
3. Espacio comprimido para MoE
4. Interpretabilidad potencial

## 2. MIXTURE OF EXPERTS (MoE)

### Gating Network

```
Input: z (32 dims) + indicadores físicos (3 dims)
  → Linear(35→64) + BatchNorm + ReLU
  → Linear(64→32) + BatchNorm + ReLU
  → Linear(32→4) + Softmax
Output: gate_weights [0,1] para cada experto
```

### Expertos

4 redes especializadas:
```
Expert i (input: 32 latent dims):
  → Linear(32→64) + BatchNorm + LeakyReLU
  → Linear(64→128) + BatchNorm + LeakyReLU
  → Linear(128→64) + BatchNorm + LeakyReLU
  → Linear(64→16)
  → Output: 16 features especializadas
```

### Mezcla

```
output = Σ(gate_weights[i] * expert_i_output) para i=0..3
```

### Indicadores Físicos Utilizados

1. **shock_indicator**: Detecta presencia de choque
2. **separation_risk**: Riesgo de separación
3. **mach_local**: Número de Mach local

Estos provienen de los features derivados (posiciones 9-11 del input de 19 features).

## 3. SENSOR VIRTUAL

### Heads Especializados

Basados en espacio latente (32 dims):

#### Shock Classifier
```
Input: z (32)
  → Linear(32→16) + BatchNorm + ReLU
  → Linear(16→8) + BatchNorm + ReLU
  → Linear(8→1) + Sigmoid
Output: P(shock) ∈ [0,1]
```

#### Intensity Regressor
```
Input: z (32)
  → Linear(32→16) + BatchNorm + ReLU
  → Linear(16→8) + BatchNorm + ReLU
  → Linear(8→1) + ReLU
Output: intensidad ≥ 0
```

#### Separation Classifier
```
Input: z (32)
  → Linear(32→16) + BatchNorm + ReLU
  → Linear(16→8) + BatchNorm + ReLU
  → Linear(8→1) + Sigmoid
Output: P(separation) ∈ [0,1]
```

## 4. FEATURES DERIVADOS (10 adicionales)

Calculados a partir de X original (9) e Y (4):

| # | Nombre | Cálculo | Dimensión |
|---|--------|---------|-----------|
| 1 | M_local | Isentrópico de Cp | [0, 5] |
| 2 | grad_P | Gradiente de presión | [-∞, ∞] |
| 3 | Cp_loss | P_isentropic - P_real | [0, 1] |
| 4 | shock_indicator | Combinado: 0.5*grad_norm + 0.3*loss_norm + 0.2*pi_norm | [-3, 3] |
| 5 | Cf_magnitude | √(Cfx² + Cfy² + Cfz²) | [0, 0.01] |
| 6 | q_dynamic | Presión dinámica isentrópica | [0, 1] |
| 7 | Pi_normalized | Pi / (1 + (γ-1)/2 * M²) | [0, 1] |
| 8 | AoA_normalized | AoA / (M + ε) | [-5, 5] |
| 9 | grad_Cf | Gradiente de fricción | [-∞, ∞] |
| 10 | L_factor | Factor Laitone de compresibilidad | [0, 1] |

## 5. DATOS Y NORMALIZACIÓN

### Tamaños de Datasets

```
Train:  3.66M samples (5% de 81.3M para demo)
Val:    406k samples
Test:   4.07M samples (10% de 40.6M para demo)
```

### Normalización

```
X_normalized = (X - X_mean) / X_std
Y_normalized = (Y - Y_mean) / Y_std

Estadísticas calculadas en train set
Aplicadas uniformemente a val y test
```

## 6. TRAINING LOOP

### Fases

**Phase 1: Autoencoder**
- Objetivo: Minimizar error de reconstrucción
- Loss: MSE ponderado por ∇P
- Epochs: 20 (demo) → 50-100 (producción)
- Early stopping: Si val_loss no mejora por 5 epochs

**Phase 2: MoE**
- Objetivo: Asignar puntos a regímenes correctos
- Loss: MSE sobre predicciones de expertos
- Epochs: 10 (demo) → 25-50 (producción)
- Encoder congelado

**Phase 3: Sensor**
- Objetivo: Predicción de choque/separación/intensidad
- Loss: BCE (shock) + MSE (intensidad)
- Epochs: Corto (5-10)
- Encoder + MoE congelados

### Validación

Cada 5 epochs:
- Evalúa en validation set
- Guarda mejor modelo si improve
- Early stopping si no mejora

## 7. EVALUACIÓN

### Métricas

```
MSE = (1/n) Σ (y_true - y_pred)²
RMSE = √MSE
MAE = (1/n) Σ |y_true - y_pred|
R² = 1 - (SS_res / SS_tot)

SS_res = Σ (y_true - y_pred)²
SS_tot = Σ (y_true - y_mean)²
```

### Análisis del Espacio Latente

```
1. PCA → 2D visualization
   - Buscar clusters
   - Identificar regímenes

2. Análisis de varianza explicada
   - Primeras 5 dims: ~80% varianza
   - Estructura jerárquica

3. Distribuciones
   - Gaussiana esperada (si AE es bueno)
   - Colas pesadas = outliers
```

## 8. HIPERPARÁMETROS CLAVE

| Parámetro | Demo | Producción | Efecto |
|-----------|------|-----------|--------|
| `train_sample_fraction` | 0.05 | 0.2-0.5 | Datos usados |
| `batch_size` | 256 | 512 | Memoria/Velocidad |
| `num_epochs` | 20 | 50-100 | Tiempo de entrenamiento |
| `learning_rate` | 1e-3 | 1e-3 | Velocidad de convergencia |
| `latent_dim` | 32 | 32-64 | Complejidad latente |
| `weight_high_gradient` | 10.0 | 5-20 | Énfasis en choques |

## 9. FÓRMULAS FÍSICAS IMPLEMENTADAS

### Mach Local Isentrópico
```
M_local = √[2/(γ-1) * ((1 + (γ-1)/2 * M²)/(1 - Cp) - 1)]
```

### Pérdida de Presión de Remanso
```
Cp_isentropic = 2/(γ*M²) * [((2 + (γ-1)*M²)/(γ+1))^(γ/(γ-1)) - 1]
Cp_loss = max(Cp_isentropic - Cp_real, 0)
```

### Factor Laitone (Compresibilidad)
```
L_factor = √(1 - M²) / (1 + 0.5*(γ-1)*M²)
```

## 10. SALIDAS Y VISUALIZACIONES

### Archivos Generados

```
outputs/
├── models/
│   └── autoencoder_best.pt          (28.8K parámetros)
├── results/
│   ├── autoencoder_evaluation.txt    (métricas)
│   └── training.log                 (log completo)
└── plots/
    ├── ae_losses.png                (train vs val)
    ├── predictions_vs_truth.png      (4 subplots)
    ├── latent_space.png             (PCA 2D)
    └── reconstruction_error.png     (distribución)
```

### Métricas Reportadas

```
MSE:           0.001 - 0.01
RMSE:          0.03 - 0.1
MAE:           0.02 - 0.08
R² Cp:         0.7 - 0.95
R² Cfx:        0.5 - 0.85
R² Cfy:        0.5 - 0.85
R² Cfz:        0.5 - 0.85
```

(Valores típicos esperados en demo con 5% de datos)

---

Para más detalles, ver `README.md` y `QUICK_START.md`.
