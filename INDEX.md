# 📑 ÍNDICE DE ARCHIVOS

## 📚 DOCUMENTACIÓN (LEE ESTO PRIMERO)

| Archivo | Propósito | Cuándo leer |
|---------|-----------|-----------|
| **DELIVERY.md** | Resumen ejecutivo del proyecto | Ahora mismo ⭐ |
| **QUICK_START.md** | Guía paso a paso | Para empezar |
| **ARCHITECTURE.md** | Detalles técnicos de la arquitectura | Para entender en profundidad |
| **README.md** | Documentación completa | Referencia general |
| **COMMANDS.sh** | Comandos útiles | Para automatizar tareas |

## ⚙️ CONFIGURACIÓN

| Archivo | Propósito |
|---------|-----------|
| **config.py** | Configuración centralizada (EDITAR AQUÍ para ajustar hiperparámetros) |
| **requirements.txt** | Dependencias Python |

## 🚀 SCRIPTS PRINCIPALES (EJECUTAR ESTOS)

| Archivo | Propósito | Comando |
|---------|-----------|---------|
| **main_train.py** | Pipeline completo de entrenamiento | `python3 main_train.py` |
| **preprocess_data.py** | Calcula features derivados (YA EJECUTADO ✅) | `python3 preprocess_data.py` |
| **infer.py** | Inferencia y análisis de resultados | `python3 infer.py --analyze` |
| **test_imports.py** | Verifica instalación | `python3 test_imports.py` |

## 📦 CÓDIGO MODULAR (src/)

| Archivo | Propósito | Responsabilidad |
|---------|-----------|-----------------|
| **src/data_loader.py** | Carga datos | Maneja 81M puntos con memory mapping y sampling inteligente |
| **src/preprocessing.py** | Features derivados | Calcula 10 variables derivadas significativas |
| **src/models.py** | Arquitectura neural | Autoencoder + MoE + Sensor Virtual (114K parámetros) |
| **src/training.py** | Loops de entrenamiento | AE trainer, MoE trainer, Sensor trainer con early stopping |
| **src/evaluation.py** | Evaluación y visualización | Métricas, plots, análisis |
| **src/__init__.py** | Package | Marca src/ como paquete Python |

## 📊 DATOS (data/)

| Archivo | Tamaño | Propósito | Creado por |
|---------|--------|-----------|-----------|
| X_train.npy | 81.3M × 9 | Features originales (entrenamiento) | Original |
| X_test.npy | 40.6M × 9 | Features originales (test) | Original |
| Ytrain.npy | 81.3M × 4 | Outputs aerodinámicos (entrenamiento) | Original |
| Ytest.npy | 40.6M × 4 | Outputs aerodinámicos (test) | Original |
| dataset.csv | 468 × 9 | Metadatos de simulaciones | Original |
| **X_train_derived.npy** | 81.3M × 19 | Features originales + 10 derivados (train) | **preprocess_data.py** ✅ |
| **X_test_derived.npy** | 40.6M × 19 | Features originales + 10 derivados (test) | **preprocess_data.py** ✅ |

## 📈 OUTPUTS (generados después de entrenar)

| Carpeta | Contenido |
|---------|-----------|
| **outputs/models/** | Modelos entrenados (.pt) |
| **outputs/results/** | Métricas y logs |
| **outputs/plots/** | Visualizaciones (PNG) |

---

## 🎯 FLUJO DE TRABAJO RECOMENDADO

### 1️⃣ **PRIMERO**: Leer documentación (5 min)
```
DELIVERY.md → Entender qué se ha hecho
QUICK_START.md → Instrucciones de ejecución
```

### 2️⃣ **SEGUNDO**: Preparar ambiente (5 min)
```bash
source .venv/bin/activate
python3 test_imports.py
```

### 3️⃣ **TERCERO**: Ejecutar entrenamiento (1-2 horas)
```bash
python3 main_train.py
# Se ejecuta automáticamente:
# - Stage 1: Carga datos
# - Stage 2: Preprocesa (features derivados)
# - Stage 3: Entrena Autoencoder
# - Stage 4: Entrena MoE
# - Stage 5: Evaluación
# - Stage 6: Visualizaciones
```

### 4️⃣ **CUARTO**: Analizar resultados (5 min)
```bash
python3 infer.py --analyze --samples 100
# Ver estadísticas del espacio latente
```

### 5️⃣ **QUINTO**: Recopilar para paper
```
Ir a outputs/plots/ y descargar figuras
Leer outputs/results/ para métricas
```

---

## 📋 ARCHIVOS POR TIPO

### Python Scripts (.py)
- **main_train.py** - Entry point principal
- **preprocess_data.py** - Preprocesamiento offline
- **infer.py** - Inferencia
- **test_imports.py** - Verificación rápida
- **config.py** - Configuración centralizada
- **src/*.py** - 5 módulos de código

**Total**: 11 archivos Python

### Documentation (.md)
- **DELIVERY.md** - Resumen ejecutivo
- **QUICK_START.md** - Guía rápida
- **ARCHITECTURE.md** - Detalles técnicos
- **README.md** - Documentación completa

**Total**: 4 archivos Markdown

### Configuration & Other
- **requirements.txt** - Dependencias
- **COMMANDS.sh** - Scripts útiles

**Total**: 2 archivos

---

## 🔍 BÚSQUEDA RÁPIDA

### Si necesitas...

**...entender cómo funciona la arquitectura**  
→ `ARCHITECTURE.md`

**...instrucciones de ejecución paso a paso**  
→ `QUICK_START.md`

**...cambiar hiperparámetros**  
→ `config.py`

**...ver cómo se cargan los datos**  
→ `src/data_loader.py`

**...entender el cálculo de features derivados**  
→ `src/preprocessing.py`

**...ver la arquitectura neural**  
→ `src/models.py`

**...ver cómo se entrena**  
→ `src/training.py`

**...entender las métricas**  
→ `src/evaluation.py`

**...una guía rápida**  
→ `QUICK_START.md`

**...comandos útiles**  
→ `COMMANDS.sh`

**...resumen ejecutivo**  
→ `DELIVERY.md`

---

## 📞 ORDEN DE LECTURA RECOMENDADO

```
1. DELIVERY.md          (5 min)  - Qué se ha hecho
2. QUICK_START.md       (5 min)  - Cómo ejecutar
3. config.py            (2 min)  - Parámetros
4. main_train.py        (1 min)  - Ver estructura
5. ARCHITECTURE.md      (10 min) - Detalles técnicos
6. src/models.py        (5 min)  - Código de modelos
```

---

## ✅ STATUS

| Componente | Status | Archivo |
|-----------|--------|---------|
| Data preprocessing | ✅ Hecho | preprocess_data.py |
| Data loading | ✅ Listo | src/data_loader.py |
| Feature engineering | ✅ Listo | src/preprocessing.py |
| Model architecture | ✅ Listo | src/models.py |
| Training pipeline | ✅ Listo | src/training.py |
| Evaluation | ✅ Listo | src/evaluation.py |
| Main script | ✅ Listo | main_train.py |
| Configuration | ✅ Listo | config.py |
| Documentation | ✅ Completa | *.md |

**TODO**: Ejecutar entrenamiento (`python3 main_train.py`)

---

## 🚀 PRÓXIMA ACCIÓN

1. Abre **DELIVERY.md**
2. Sigue los pasos en **QUICK_START.md**
3. Ejecuta: `python3 main_train.py`
4. Espera resultados
5. Analiza con: `python3 infer.py --analyze`

¡Listo para presentar el paper! 📄
