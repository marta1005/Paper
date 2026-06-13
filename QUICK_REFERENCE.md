# ⚡ QUICK REFERENCE CARD

## 🚀 EMPEZAR AHORA (5 minutos)

```bash
# 1. Activar ambiente
source .venv/bin/activate

# 2. Verificar que funciona
python3 test_imports.py

# 3. ENTRENAR (MAIN COMMAND)
python3 main_train.py

# 4. Esperar a que termine (~1-2 horas en CPU)

# 5. Ver resultados
python3 infer.py --analyze --samples 100
```

---

## 📊 DATOS EN UN VISTAZO

| Métrica | Valor |
|---------|-------|
| Train samples | 3.66M (demo, 5% de 81.3M total) |
| Test samples | 4.07M (demo, 10% de 40.6M total) |
| Input features | 19 (9 orig + 10 derivados) |
| Output features | 4 (Cp, Cfx, Cfy, Cfz) |
| Latent dimension | 32 |
| Model parameters | 114K |

---

## 🏗️ ARQUITECTURA EN 30 SEGUNDOS

```
Input (19)
   ↓
[AE: 19→128→64→32→64→128→19]
   ↓
Latent (32)
   ↓
[MoE: 4 Expertos + Gating]
   ↓
Salidas: shock_prob, intensidad, separation_prob
```

---

## 🔧 CAMBIAR HIPERPARÁMETROS

Editar `config.py`:

```python
# Línea ~19: Más datos para mejores resultados
'train_sample_fraction': 0.2,  # De 0.05 → 0.2 (20% = 16M puntos)

# Línea ~28: Más épocas
'num_epochs': 50,  # De 20 → 50

# Línea ~27: Batch size mayor
'batch_size': 512,  # De 256 → 512

# Línea ~26: Learning rate
'learning_rate': 1e-3,  # Probar 1e-4 a 1e-2
```

Luego: `python3 main_train.py`

---

## 📁 ARCHIVOS CLAVE

| Archivo | Editar? | Para qué |
|---------|---------|----------|
| **config.py** | ✅ SÍ | Cambiar hiperparámetros |
| **main_train.py** | ❌ No | Ejecutar training |
| **src/models.py** | ❌ No | Ver arquitectura |
| **src/data_loader.py** | ❌ No | Entender cómo cargan datos |

---

## 📖 DOCUMENTACIÓN

| Archivo | Para |
|---------|------|
| **DELIVERY.md** | Qué se hizo (LEER PRIMERO) |
| **QUICK_START.md** | Cómo ejecutar paso a paso |
| **ARCHITECTURE.md** | Detalles técnicos y fórmulas |
| **README.md** | Documentación completa |
| **COMMANDS.sh** | Comandos útiles |

---

## 🐛 PROBLEMAS COMUNES

| Problema | Solución |
|----------|----------|
| "File not found" | `pwd` debería ser `/Users/martaarnabatmartin/Desktop/Paper` |
| "Out of memory" | Reducir `batch_size` a 128 en config.py |
| "Module not found" | Ejecutar `python3 test_imports.py` para diagnosticar |
| Muy lento | Está bien en CPU, esperar o usar GPU |

---

## ✅ CHECKLIST

- [ ] Leer DELIVERY.md (5 min)
- [ ] Setup: `source .venv/bin/activate`
- [ ] Test: `python3 test_imports.py`
- [ ] Entrenar: `python3 main_train.py` (1-2 horas)
- [ ] Analizar: `python3 infer.py --analyze`
- [ ] Copiar figuras de `outputs/plots/`
- [ ] Escribir paper con resultados

---

## 💾 ARCHIVOS GENERADOS

```
outputs/
├── models/autoencoder_best.pt
├── results/autoencoder_evaluation.txt
├── results/training.log
└── plots/
    ├── ae_losses.png
    ├── predictions_vs_truth.png
    ├── latent_space.png
    └── reconstruction_error.png
```

---

## 🎓 PARA EL PAPER

### Sección Metodología
- Describir: AE (compresión) + MoE (regímenes) + Sensor (predicción)
- Mostrar: Diagrama de arquitectura
- Explicar: 10 variables derivadas

### Sección Resultados
- Incluir: Figuras de `outputs/plots/`
- Reportar: RMSE, MAE, R² de `outputs/results/evaluation.txt`
- Analizar: Espacio latente y clusters

### Sección Conclusiones
- Discutir: Capacidad de generalización
- Limitaciones: Solo puntos, sin topología
- Trabajo futuro: GNN, pySR, validación experimental

---

## 🚦 TIEMPO ESTIMADO

| Tarea | CPU | GPU |
|-------|-----|-----|
| Setup | 5 min | 5 min |
| Preprocess datos | 15 min | 15 min |
| Entrenar AE | 30 min | 5 min |
| Entrenar MoE | 15 min | 2 min |
| Evaluar | 5 min | 1 min |
| **TOTAL** | **~1.5 h** | **~10 min** |

---

## 📞 SOPORTE RÁPIDO

```bash
# Ver config
cat config.py | grep "train_sample"

# Ver logs en tiempo real
tail -f outputs/training.log

# Contador de archivos
find src -name "*.py" | wc -l

# Tamaño de datos
du -sh data/

# Archivos más grandes
ls -lhS | head -5
```

---

## 🎯 MÉTRICA DE ÉXITO

El modelo está bien entrenado si:

- ✅ RMSE < 0.05
- ✅ R² > 0.7
- ✅ Latent space muestra clusters
- ✅ MoE asigna puntos a regímenes correctos
- ✅ Gráficas de `outputs/plots/` son interpretables

---

## 🔗 COMANDOS RÁPIDOS

```bash
# Entrenar
python3 main_train.py

# Analizar (después de entrenar)
python3 infer.py --analyze --samples 100

# Ver logs
tail -f outputs/training.log

# Monitor
top -n 1 | head -10

# Limpiar caché
find . -type d -name __pycache__ -delete
```

---

## 💡 PRO TIPS

1. **Primero demo (5%)** → Verificar que funciona  
   Luego **aumentar a 20%** → Mejores resultados

2. **Entrenamiento en background**: `nohup python3 main_train.py > log.txt 2>&1 &`

3. **Monitor de RAM**: `top -u $(whoami)` durante entrenamiento

4. **Guardar output**: `python3 main_train.py | tee output.log`

5. **Comparar modelos**: Guardar `config.py` con cada experimento

---

## 📚 REFERENCIA RÁPIDA: VARIABLES DERIVADAS

```
1. M_local       : Mach isentrópico
2. grad_P        : Gradiente de presión
3. Cp_loss       : Pérdida de presión de remanso
4. shock_indicator: Indicador de choque (combinado)
5. Cf_magnitude  : Magnitud de fricción
6. q_dynamic     : Presión dinámica
7. Pi_normalized : Presión normalizada
8. AoA_normalized: Ángulo normalizado
9. grad_Cf       : Gradiente de fricción
10. L_factor     : Factor de compresibilidad
```

**Input original**: 9 features  
**Input procesado**: 19 features (9 + 10 derivados)

---

**LISTO PARA EMPEZAR?** 

Ejecuta: `python3 main_train.py`

¡Éxito! 🚀
