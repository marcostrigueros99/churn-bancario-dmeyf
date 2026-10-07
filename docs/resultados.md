# Resumen de Resultados — Competencia 1

## Resultados públicos (bot Zulip)

| Submit | Fecha | Modelo | Public gain mean | Std dev |
|---|---|---|---|---|
| v1 | 2026-10-06 | v3 semillerio | 83.85M | 3.36 |
| v2 | 2026-10-06 | v4 exp B/C | 85.85M | 1.66 |
| v3 | 2026-10-07 | v5 kitchen sink | **87.07M** | 2.63 |

## Walk-Forward fold_4 por versión

| Versión | Ganancia | AUC | Threshold | N intervenidos |
|---|---|---|---|---|
| v2 (bug) | $420.420.000 | 0.8991 | 0.533 | 16.112 |
| v3 | $377.685.000 | 0.8631 | 0.153 | 9.866 |
| v4 | ~$377.685.000 | 0.8631 | — | — |
| v5 | ~$396.275.000 | — | — | — |

## Análisis de errores (FN — churners perdidos, fold_4 v3)

| Variable | TP (detectados) | FN (perdidos) |
|---|---|---|
| mcuentas_saldo | -$18.172 | +$77.450 |
| cliente_antiguedad | 108 meses | 116 meses |
| cproductos | 5.92 | 6.96 |
| prob predicha | 0.22 | 0.08 |

Los churners perdidos son "buenos clientes" — mayor saldo, más antigüedad, más productos.

## Frentes de mejora pendientes

- Oversampling selectivo de FN (duplicar en train los churners con perfil buen cliente)
- Target encoding de variables categóricas (Master_status, Visa_status)
- Más trials de Optuna con params v5 como warm start
- Analizar qué configuración ganó en el kitchen sink
