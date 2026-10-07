# Predicción de Churn Bancario
### Data Mining en Economía y Finanzas — Competencia 1 · 2026

Proyecto de predicción de churn bancario sobre un panel longitudinal de clientes.
El objetivo es maximizar una función de ganancia asimétrica identificando qué clientes
se darán de baja el mes siguiente al último disponible en el dataset.

---

## Descripción del problema

| Dimensión | Valor |
|---|---|
| Dataset | 983.061 filas × 155 columnas |
| Período | 202103 → 202108 (6 meses) |
| Target | `clase_ternaria`: continua, baja1, baja2, sin_target |
| Positivos (baja2) | 4.067 / 650.066 entrenables (0.626%) |
| Desbalance | 158:1 |
| Mes a predecir | 202108 |

## Función de ganancia

```
TP (baja2 detectado e intervenido): +$1.072.500
FP (intervención en cliente que no se va): -$27.500
FN / TN: $0
Ratio de equilibrio: 39:1
```

El threshold de clasificación no es 0.5 — se optimiza barriendo thresholds
y maximizando la función de ganancia directamente.

---

## Metodología

### Validación: Walk-Forward
En datos temporales no se puede usar cross-validation estándar (data leakage).
Usamos walk-forward: siempre entrenamos con datos anteriores al mes de test.

| Fold | Train | Test |
|---|---|---|
| F2 | 202103 | 202104 |
| F3 | 202103-202104 | 202105 |
| F4 ★ | 202103-202105 | 202106 |

### Modelo
LightGBM con optimización bayesiana de hiperparámetros (Optuna, TPESampler).
Métrica objetivo de Optuna: ganancia en el threshold óptimo (no AUC).

### Feature Engineering
Sobre 25 variables base del dataset:
- **Lag t-1, t-2:** valor del mes anterior y anterior al anterior
- **Delta y aceleración:** variación absoluta y variación de la variación
- **Delta relativo:** variación proporcional (clippeada a ±10)
- **Ventana 3 meses:** media, desvío estándar y coeficiente de variación
- **Tendencia:** pendiente de regresión lineal sobre 3 meses
- **Ratios de portfolio:** rentabilidad por producto, digitalización relativa,
  comisiones sobre rentabilidad, saldo por antigüedad, utilización de tarjetas,
  margen neto, endeudamiento relativo
- **Flags:** primer mes en el panel, tiene Visa, tiene Master, tiene préstamo

Total: **~327 columnas** (dataset guardado en `dataset_fe_v3.parquet`)

### Submission: Semillerio
Para cada submit generamos 20 archivos: 4 semillas × 5 cortes alrededor del
N óptimo del fold_4. El bot devuelve la ganancia media y el desvío entre los 20 modelos.
Desvío bajo = modelo estable.

---

## Resultados

### Walk-Forward (fold_4)

| Versión | Ganancia fold_4 | AUC | Notas |
|---|---|---|---|
| v2 (baseline) | $420M | 0.899 | Bug en Optuna — árbol trivial |
| v3 | $377M | 0.863 | Fix Optuna, FE extendido |
| v4 (spw agresivo) | ~$377M | 0.863 | scale_pos_weight 2x-5x el natural |
| v5 (kitchen sink) | ~$396M | — | Optuna con espacio completo |

### Resultados públicos (competencia)

| Submit | Modelo | Public gain mean | Std dev | Mejora |
|---|---|---|---|---|
| v1 | v3 semillerio | 83.85M | 3.36 | — |
| v2 | v4 exp B/C | 85.85M | 1.66 | +2.00M |
| v3 | v5 kitchen sink | **87.07M** | 2.63 | +1.22M |

> Los valores públicos corresponden al 25% del dataset evaluado por el bot.
> Escalado al 100%: ~348M de ganancia estimada.

---

## Estructura del repositorio

```
churn-bancario-dmeyf/
├── notebooks/
│   ├── 01_eda_churn_bancario.ipynb
│   ├── 02_calidad_datos_churn.ipynb
│   ├── 03_construccion_target.ipynb
│   ├── 04_fe_modelo_optuna_v3.ipynb
│   ├── 05_submission_semillerio.ipynb
│   ├── 06_modelo_v4_submision.ipynb
│   ├── 07_experimentos_v5.ipynb
│   └── 08_setup_github.ipynb
├── params/
│   ├── mejores_params_optuna_v3.json
│   ├── mejores_params_optuna_v4.json
│   └── mejores_params_optuna_v5.json
├── metricas/
│   └── metricas_experimentos.csv
├── docs/
│   └── documentacion_proyecto_churn.md
├── .gitignore
└── README.md
```

> Los archivos de datos (`.parquet`, `.csv`) no se incluyen en el repo
> por tamaño — se mantienen en Google Drive.

---

## Hiperparámetros optimizados (Optuna)

| Parámetro | Rango | Justificación |
|---|---|---|
| `num_leaves` | [16, 256] | Control principal de complejidad |
| `max_depth` | [4, 12] | Regularización adicional |
| `learning_rate` | [0.01, 0.3] log | Velocidad de aprendizaje |
| `n_estimators` | [100, 1000] | Número de árboles |
| `min_child_samples` | [20, 2000] log | Mínimo por hoja — crítico con 158:1 |
| `min_split_gain` | [0.0, 1.0] | Ganancia mínima para split |
| `feature_fraction` | [0.4, 1.0] | % features por árbol |
| `bagging_fraction` | [0.4, 1.0] | % muestras por árbol |
| `reg_alpha` | [1e-8, 10] log | L1 regularización |
| `reg_lambda` | [1e-8, 10] log | L2 regularización |
| `scale_pos_weight` | [spw/4, spw×5] | Peso del desbalance |
| `boosting_type` | gbdt/dart/goss | Tipo de boosting |
| `n_meses_train` | [1, 3] | Ventana de entrenamiento |

---

## Tecnologías

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green)
![Optuna](https://img.shields.io/badge/Optuna-3.x-orange)
![Colab](https://img.shields.io/badge/Google_Colab-F9AB00?logo=googlecolab)

- **Modelo:** LightGBM
- **Optimización:** Optuna (TPE Sampler)
- **Entorno:** Google Colab + Google Drive
- **Almacenamiento:** Parquet (checkpoints), JSON (params), CSV (métricas)
