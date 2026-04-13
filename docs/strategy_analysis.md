# Análisis Estratégico: Modal, Flash Attention y Modelos Pequeños

## 1. Flash Attention — Ya lo tienes (parcialmente)

Buenas noticias: **tu código ya usa Flash Attention**.

```python
# model.py línea 158
output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask, dropout_p=0.0, is_causal=False)
```

`F.scaled_dot_product_attention` es la API nativa de PyTorch 2.x que **despacha automáticamente al kernel FlashAttention-2** cuando se cumplen las condiciones:
- ✅ Backend CUDA (GPUs NVIDIA)
- ✅ Dtype FP16 o BF16
- ✅ Sin máscara custom (usar `is_causal=True`)

El problema es que **en DirectML (tu Radeon 780M) se ejecuta la implementación matemática naive** — no existe kernel Flash Attention para AMD iGPUs. Los "bugs" que encontrasteis fueron probablemente incompatibilidades de DirectML, no del algoritmo en sí.

> [!IMPORTANT]
> **En Modal con CUDA, Flash Attention ya funciona sin cambios de código.** Solo hay un ajuste menor recomendado: cambiar `is_causal=False` + máscara manual → `is_causal=True` sin máscara. Esto permite que PyTorch use el kernel más optimizado.

### Ajuste recomendado para máximo rendimiento CUDA

En el forward de `Attention`, cuando `seqlen > 1` y no hay KV-cache:
```python
# En lugar de construir la máscara manualmente y usar is_causal=False:
output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=0.0, is_causal=True)

# Solo pasar mask cuando hay KV-cache (inference incremental):
output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask, dropout_p=0.0, is_causal=False)
```

Esto evita construir la máscara O(n²) y habilita el path más rápido del kernel.

---

## 2. Modal.com — Análisis de $30/mes

### Precios actuales por GPU

| GPU | VRAM | $/hora | Horas con $30 | TFLOPS FP16 |
|-----|------|--------|---------------|-------------|
| **T4** | 16 GB | $0.59 | **50.8h** | ~65 |
| **L4** | 24 GB | $0.80 | **37.5h** | ~121 |
| **A10G** | 24 GB | $1.10 | **27.3h** | ~125 |
| **A100 40GB** | 40 GB | $2.10 | **14.3h** | ~312 |

### Estimación de velocidad vs. tu Radeon 780M

Tu entrenamiento actual: **~93s/iter** (batch=4, seq=512, accum=4, **FP32 sin tensor cores**).

En CUDA con BF16 + tensor cores + Flash Attention nativo, modelos de este tamaño son **30-100x más rápidos**:

| Hardware | Est. tiempo/iter | 10,000 iters | Coste |
|----------|-----------------|--------------|-------|
| **Radeon 780M** (actual) | ~93s | **~10.7 días** | Gratis |
| **T4** (CUDA FP16) | ~4-5s | **~12.5h** | ~$7.40 |
| **L4** (CUDA BF16) | ~2-3s | **~7h** | ~$5.60 |
| **A10G** (CUDA BF16) | ~1.5-2.5s | **~5.5h** | ~$6.05 |

> [!TIP]
> **La L4 es probablemente el sweet spot.** Más rápida que la T4, más barata por entrenamiento completado que la A10G, y 24GB de VRAM permiten batch sizes mayores. Un entrenamiento completo de 78M costaría **~$5-6** de tus $30.

### ¿Qué puedes hacer con $30?

| Escenario | Coste est. | Qué consigues |
|-----------|-----------|---------------|
| 1x entrenamiento 78M completo (10k iters) | ~$6 | Validar pipeline V1 en horas en vez de días |
| 5x sweep modelos pequeños (10-30M) | ~$5-8 | Encontrar hiperparámetros óptimos |
| 1x entrenamiento 78M + dataset ampliado (V2) | ~$12-15 | Modelo con más tokens, mejor convergencia |
| Combo: sweeps pequeños + 1x final 78M | ~$15-20 | **Estrategia óptima** |

---

## 3. Modelos Pequeños — ¿Tiene sentido?

**Sí, absolutamente.** Es la decisión más inteligente que puedes tomar ahora mismo.

### El argumento de las Scaling Laws

Con tu dataset de 108M tokens:

| Modelo | Params | Ratio tokens:params | Estado |
|--------|--------|---------------------|--------|
| 78M | 78.13M | **1.38:1** | Severamente sub-entrenado |
| 30M | ~30M | **3.6:1** | Sub-entrenado pero funcional |
| 20M | ~20M | **5.4:1** | Razonable para experimentos |
| 10M | ~10M | **10.8:1** | Cercano a medio-Chinchilla |

> [!NOTE]
> Chinchilla-óptimo es 20:1, pero la industria moderna sobre-entrena masivamente (Llama 3 usa 1875:1). Para tus fines experimentales, un ratio de **5-10:1 es perfectamente válido** para obtener un modelo funcional que genere texto coherente.

### Qué ganas con modelos pequeños

1. **Iteración rápida**: Un modelo de 10M entrena en **~1h en L4** (~$0.80). Puedes probar 10 configs diferentes por $8.
2. **Validación del pipeline**: Confirmar que el dataset, tokenizador, curriculum y formato `<think>` funcionan correctamente ANTES de gastar recursos en el 78M.
3. **Búsqueda de hiperparámetros**: LR, warmup, weight decay, seq_len... cada uno afecta dramáticamente la convergencia. Es absurdo buscarlos en un modelo que tarda 10 días.
4. **Ablation studies**: ¿Cuánto mejora el dato sintético de lógica vs. solo FineWeb-Edu? Lo puedes medir con modelos de 10M en minutos.
5. **Baseline comparativo**: Cuando entrenes el 78M con los hiperparámetros optimizados, tendrás una curva de referencia para saber si escala correctamente.

### Configs recomendadas para modelos pequeños

```yaml
# --- TinyThinker 10M (Nano) ---
dim: 192
n_layers: 6
n_heads: 6
n_kv_heads: 3
vocab_size: 16384
max_seq_len: 512
# Params estimados: ~10.5M
# Ratio con 108M tokens: 10.3:1

# --- TinyThinker 20M (Micro) ---
dim: 256
n_layers: 8
n_heads: 8
n_kv_heads: 4
vocab_size: 16384
max_seq_len: 512
# Params estimados: ~20M
# Ratio con 108M tokens: 5.4:1

# --- TinyThinker 30M (Mini) ---
dim: 320
n_layers: 10
n_heads: 8
n_kv_heads: 4
vocab_size: 16384
max_seq_len: 512
# Params estimados: ~30M
# Ratio con 108M tokens: 3.6:1
```

---

## 4. Estrategia Recomendada

### Fase A — Sweeps locales rápidos (GRATIS, Radeon 780M)
Mientras el 78M entrena de fondo:
1. Crear config YAML para el modelo de **10M**
2. Entrenar 2000 iters localmente (~5-6 horas con el 10M)
3. Verificar que genera texto coherente y respeta `<think>`

### Fase B — Sweeps en Modal ($8-10)
1. Subir `train_v1.bin` al volumen de Modal
2. Correr **3-4 entrenamientos del 10M** variando LR y warmup
3. Correr **1 entrenamiento del 20M** con los mejores hiperparámetros encontrados
4. Evaluar: ¿el 20M es proporcionalmente mejor que el 10M?

### Fase C — Entrenamiento final en Modal ($6-8)
1. Aplicar los hiperparámetros óptimos descubiertos en Fase B
2. Entrenar el **78M completo** en L4/A10G (5-7 horas)
3. Con Flash Attention nativo + BF16, será dramáticamente más rápido

### Fase D — Expansión (si queda presupuesto)
1. Ampliar dataset a V2 (más FineWeb-Edu + Cosmopedia)
2. Re-entrenar el 78M con más datos

> [!IMPORTANT]
> **Sobre el script `modal_train.py` actual:** Existe pero necesita actualizarse. Apunta a `train_combined.bin` (legacy) en vez de `train_v1.bin`, no usa `--config`, y los argumentos hardcodeados no reflejan tu setup V1. Habría que parchearlo antes de usarlo.

---

## 5. Sobre el entrenamiento actual en la Radeon 780M

**No lo pares.** Aunque Modal sea más rápido, el entrenamiento local tiene valor:
- Cada iteración que completa es progreso gratuito
- Si preparas Modal bien, puedes usar el checkpoint local como punto de partida (`--resume`) en Modal para no desperdiciar el trabajo hecho
- Cuando el de Modal termine, tendrás dos puntos de comparación

### Dato clave sobre tu entrenamiento actual
El warmup está en 200 iters (hardcodeado) en vez de las 1000 del YAML. Esto significa que la LR ya llegó a su máximo (1e-4) en la iter 200. No es catastrófico — el modelo está aprendiendo — pero si decides relanzar desde cero en Modal, sería bueno corregir ese bug primero para que respete el YAML.
