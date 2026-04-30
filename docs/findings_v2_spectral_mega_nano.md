# Findings v2 — SpectralThinker Mega-Nano (dim=1024, k=64)

**Fecha:** 2026-04-29 / 2026-04-30
**Duración:** ~9h52m (CPU, 3000 iters)
**Config:** `configs/train_spectral_mega_nano.yaml`

---

## Configuración

| Parámetro | Valor |
|-----------|-------|
| `dim` | 1024 |
| `n_layers` | 8 |
| `k_dim_attn` / `k_dim_ffn` | 64 / 64 (igual que Nano) |
| `batch_size` | 8, `grad_accum` 8 → 16,384 tokens/iter |
| `max_iters` | 3000 (vs 5000 del Nano) |
| Tokens entrenados | 49.2M (vs 81.9M del Nano) |

## Conteo de parámetros

| Componente | Params |
|-----------|--------|
| Embedding + output (sin weight tying efectivo) | 2 × 16,777,216 = 33.55M |
| Núcleos espectrales (attn + FFN, 8 capas) | 327,680 |
| **Total entrenables** | **17,122,304** |
| Compresión de proyecciones vs denso equiv. | **~256x** (vs 34x del Nano) |

> Nota crítica: weight tying roto (igual que en Nano). Embedding cuenta 2x en total params
> pero solo hay 1 copia del tensor. Tokens/parámetros efectivos: 49.2M / 17M = 2.9 — muy bajo.

---

## Curva de convergencia

| Iter | Val Loss | Perplexity | Δ |
|------|----------|-----------|---|
| 250  | 5.8159   | 335       | — |
| 500  | 4.6366   | 103       | -1.18 |
| 2250 | **3.9021** | **49** | — ← MEJOR |
| 2500 | 3.9894   | 54        | +0.09 ← overfitting |
| 2750 | 4.0194   | 56        | +0.03 ← peor |

---

## Comparativa con Nano

| Modelo | dim | k | Proj. params | Mejor val_loss | Perplexity |
|--------|-----|---|-------------|----------------|-----------|
| Nano | 256 | 64 | 245K | **3.7318** | **42** |
| Mega-Nano | 1024 | 64 | 328K | 3.9021 | 49 |

El Nano gana en resultado final, pero el Mega-Nano converge más rápido inicialmente
(val_loss 4.64 en iter 500 vs 5.01 del Nano en el mismo punto).

---

## Diagnóstico: el problema no son los núcleos, es el embedding

Los núcleos espectrales k=64 funcionan perfectamente a dim=1024 (256x de compresión).
El problema es el embedding de 16.78M params que necesita muchos más tokens para generalizar:

- Nano: 81.9M tokens / 4.19M embedding = **19.5 tokens por param de embedding**
- Mega-Nano: 49.2M tokens / 16.78M embedding = **2.9 tokens por param de embedding**

El overfitting se manifiesta a partir de iter 2250 (75% del entrenamiento), cuando el
embedding memoriza el training set y val_loss empieza a subir.

**El fallo NO es la arquitectura espectral. Es la relación datos/embedding.**

---

## Lecciones aprendidas

1. **k=64 escala a dim=1024** — la compresión 256x es viable (validado hasta iter 2250)
2. **Weight tying roto** — aumenta artificialmente el conteo de params y complica el embedding
3. **El embedding domina** — a dim=1024 con vocab=16384, el embedding es el cuello de botella
4. **Estrategia de escala correcta**: aumentar k antes que dim si se quiere más capacidad sin inflar el embedding

---

## Conclusión

La hipótesis espectral se mantiene. Los coeficientes DCT/Walsh k=64 funcionan en espacios
de dimensión 1024. El resultado final fue limitado por la relación datos/embedding,
no por la capacidad espectral de las proyecciones.

**Próximo paso prioritario**: SpectralThinker Midi (dim=512, k=128) con weight tying correcto,
que ofrece mejor ratio datos/params y mayor compresión por núcleo k.
