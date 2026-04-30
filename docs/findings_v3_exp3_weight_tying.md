# Findings v3 — EXP-3: Weight Tying en SpectralThinker Nano

**Fecha:** 2026-04-30
**Comparativa:** Nano v1 (tying roto = matrices separadas) vs Nano v2 (tying correcto)

---

## Resultados

| Modelo | Params | Weight Tying | Mejor val_loss | Perplexity | Iter mejor |
|--------|--------|-------------|----------------|-----------|-----------|
| Nano v1 | 8.64M | ❌ Separadas | **3.7318** | **42** | 4250 |
| Nano v2 | 4.44M | ✅ Compartidas | 3.7850 | 44 | 3500 |

**El Nano v1 (matrices separadas) gana por 0.053 puntos de val_loss.**

Nano v2 también presenta mayor oscilación entre checkpoints:
- v1: curva suave, best en iter 4250, estable
- v2: oscila (3.785 → 3.870 → 3.810 → 3.887 → 3.790 → 3.862)

---

## Análisis

El weight tying fuerza que la representación de entrada de un token sea idéntica
a su vector de probabilidad de salida. Son tareas relacionadas pero distintas:
- **Input embedding:** "qué significa este token en contexto"
- **Output lm_head:** "qué probabilidad dar a este token como siguiente"

Con matrices separadas, el modelo tiene libertad para aprender representaciones
diferenciadas para cada tarea. En modelos pequeños (~4-8M params), esa libertad
compensa con creces el ahorro de parámetros.

## Conclusión

El weight tying es beneficioso en modelos de >1B params donde el ahorro de
memoria es crítico. Para SpectralThinker a escala nano/midi, las matrices
separadas dan mejor resultado. El "bug" del Nano v1 era en realidad la
configuración óptima.

**Decisión:** En todos los experimentos futuros (Midi, Large) se usarán
matrices separadas (sin weight tying forzado) hasta que el tamaño del modelo
lo justifique.
