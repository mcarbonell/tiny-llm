# Plan: DCT-LLM — LLM Espectral desde Cero

## Objetivo

Construir un LLM real entrenado desde cero cuya arquitectura interna reemplaza
todas las proyecciones densas por capas espectrales:
- **DCT** (`DCTLinear`) → proyecciones de Atención (Q, K, V, O)
- **Walsh-FWHT** (`WalshLinear`) → proyecciones del FFN (w1, w2, w3)

El punto de partida es `tiny-thinker`, que ya tiene toda la infraestructura
(tokenizador, datos, `train.py`, `chat.py`). Solo hay que añadir un nuevo
archivo de modelo y un config YAML.

---

## Preguntas abiertas / Decisiones de diseño

> [!IMPORTANT]
> **¿Cuál es el tamaño objetivo del modelo?**
> El tiny-thinker tiene configs para 10M (nano), 20M (micro) y 78M.
> Para una primera validación del DCT-LLM recomendaría empezar con **~10M params**
> en la versión espectral (equivalente a un nano) y comparar contra el nano denso.
> Si estás de acuerdo, el plan usa esa escala.

> [!IMPORTANT]
> **¿Qué `dim` usar?**
> `WalshLinear` exige que las dimensiones sean potencias de 2.
> El nano actual usa `dim=256`, que cumple el requisito.
> Si quieres escalar a `dim=512`, también es potencia de 2. Confirma cuál prefieres.

> [!NOTE]
> **RoPE en DCT-Attention**
> El prototipo V67 usaba embeddings posicionales aprendibles simples.
> La integración en tiny-thinker debería mantener RoPE (ya funciona y es mejor).
> Esto requiere adaptar el `DCTAttention` para recibir `freqs_cis` y pasar por `apply_rotary_emb`.

---

## Cambios propuestos

### `tiny-thinker/model/`

#### [NEW] `model_spectral.py`
Nuevo archivo de modelo. **No toca ningún archivo existente.**

Contenido:
1. `get_dct_matrix_1d(N)` — generador de base DCT (copiado y limpiado de V67).
2. `get_walsh_matrix_1d(N)` — generador de base Walsh con Sylvester construction (de V67).
3. `DCTLinear(in, out, k_in, k_out)` — síntesis `W = D_out^T @ C_padded @ D_in`.
4. `WalshLinear(in, out, k_in, k_out)` — ídem con base Walsh.
5. `SpectralFeedForward(dim, hidden_dim, k_dim, k_hidden)` — SwiGLU con `WalshLinear`.
6. `SpectralAttention(args)` — GQA con `DCTLinear` para Q, K, V, O + RoPE.
7. `SpectralTransformerBlock(args)` — bloque completo.
8. `SpectralArgs` — dataclass de configuración (hereda los campos de `ModelArgs`
   y añade `k_dim_attn`, `k_dim_ffn`, `k_hidden_ffn`).
9. `SpectralThinker(SpectralArgs)` — modelo completo compatible con `train.py`.

**Compatibilidad con `train.py`:**
El `train.py` existente usa duck typing sobre el modelo
(`model(tokens, targets)` devuelve `(logits, loss)`).
`SpectralThinker` implementará la misma interfaz exacta para no modificar `train.py`.

**Compatibilidad con DirectML:**
Se copiarán los workarounds del `model.py` original para `aten::embedding`
y `aten::pow.Tensor_Scalar` en la `RMSNorm`.

---

### `tiny-thinker/configs/`

#### [NEW] `train_spectral_nano.yaml`
Config YAML para el primer experimento de validación:

```yaml
# Spectral Nano — primera validación DCT+Walsh
dim: 256
n_layers: 6
n_heads: 8
n_kv_heads: 4
vocab_size: 16384
max_seq_len: 256

# Parámetros espectrales
k_dim_attn: 64      # Compresión ~4x en atención (256 -> 64)
k_dim_ffn: 64       # Compresión en FFN entrada/salida
k_hidden_ffn: 128   # Compresión en FFN dimensión oculta

# Entrenamiento
batch_size: 16
seq_len: 256
max_iters: 5000
learning_rate: 0.001
...
model_file: model/model_spectral.py
model_class: SpectralThinker
```

---

### `tiny-thinker/scripts/`

#### [MODIFY] `config.py` (mínimo)
Añadir lectura de los nuevos campos `k_dim_attn`, `k_dim_ffn`, `k_hidden_ffn`
del YAML para pasarlos a `SpectralArgs`. Cambio de ~5 líneas, no rompe nada existente.

---

## Estrategia de validación en fases

### Fase 0 — Smoke test (día 1)
- Crear `model_spectral.py` con la arquitectura completa.
- Lanzar con `train_local.yaml` adaptado, **solo 100 iteraciones**, verificar:
  - El modelo compila sin errores.
  - La loss desciende en los primeros 5 batches (regla de oro del GEMINI.md).
  - No hay errores de DirectML.

### Fase 1 — Benchmark vs. Nano denso (semana 1)
- Entrenar **Spectral Nano** 5000 iteraciones con `data/train_v1.bin`.
- Comparar curva de loss contra el Nano denso del PROJECT_STATUS.md (perplexity: 244.95).
- Objetivo: perplexity similar con **menos parámetros entrenables**.

**Métricas a registrar:**
- `total_params` (entrenables vs. totales incluyendo buffers DCT/Walsh).
- `perplexity_final`.
- `wall_clock_time_per_iter`.
- `overhead_spectral` (tiempo extra por síntesis de W en cada forward).

### Fase 2 — Optimización (semana 2)
- **Cache de W**: Las matrices `D_in` y `D_out` son fijas. La síntesis `W = D_out^T @ C @ D_in`
  se puede cachear entre iteraciones (solo recalcular cuando `C` cambia).
  Esto elimina el overhead de síntesis durante la inferencia.
- **Ajuste de K**: Barrer `k_dim_attn` ∈ {32, 64, 96} para encontrar el equilibrio
  compresión/perplexity.

### Fase 3 — Escalar (semana 3-4)
- Si Fase 1 muestra perplexity ≤ 10% peor que el nano denso con más compresión,
  escalar a `dim=512`, más capas, más datos.

---

## Estimación de parámetros entrenables (Spectral Nano)

| Componente | Dense params | Spectral params | Ratio |
|---|---|---|---|
| Atención Q,K,V,O (×6 capas) | 6 × 4 × 256² = 1.57M | 6 × 4 × 64² = 98K | **~16x** |
| FFN w1,w2,w3 (×6 capas) | 6 × 3 × 256×512 = 2.36M | 6 × 3 × 64×128 = 147K | **~16x** |
| Embeddings + output | 2 × 16384×256 = 8.39M | 8.39M (sin cambio) | 1x |
| **Total** | ~12.3M | ~8.6M | — |
| **Solo proyecciones** | 3.93M | 245K | **~16x** |

> [!NOTE]
> Los embeddings de vocabulario dominan el recuento total porque son 16K × 256.
> El beneficio espectral es máximo en las proyecciones de atención y FFN,
> que son exactamente las partes que escalan cuadráticamente al aumentar `dim`.
> En un modelo de `dim=512` la ganancia es 4x mayor en términos absolutos.

---

## Lo que NO cambia

- `train.py` — sin modificaciones.
- `chat.py` — sin modificaciones (carga el modelo por `model_file` dinámicamente).
- `tokenizer.json` / `tokenizer_v1.json` — sin tocar.
- Todos los modelos existentes (`model.py`, `model_coga.py`, `model_dense.py`) — intactos.
- Los datos (`train_v1.bin`) — los mismos.

---

## Orden de implementación

1. `model/model_spectral.py` — el bloque central.
2. `configs/train_spectral_nano.yaml` — config de experimento.
3. `scripts/config.py` — añadir lectura de campos espectrales.
4. Smoke test (tú ejecutas).
5. Entrenamiento Fase 1 (tú ejecutas, yo analizo los logs).
