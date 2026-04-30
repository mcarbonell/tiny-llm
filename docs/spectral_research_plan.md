# Plan de Experimentos — SpectralThinker Research Roadmap

Basado en los resultados de los experimentos v1 (Nano) y v2 (Mega-Nano),
este documento define la hoja de ruta de experimentos prioritarios.

---

## Estado actual (completado)

| Exp | Modelo | dim | k | Val Loss | Perplexity | Compresión |
|-----|--------|-----|---|----------|-----------|-----------|
| v1 | Nano | 256 | 64 | **3.73** | **42** | 34x |
| v2 | Mega-Nano | 1024 | 64 | 3.90 | 49 | 256x |

**Conclusiones clave:**
- k=64 funciona a dim=256 y dim=1024
- Weight tying está roto → hay que arreglarlo
- El embedding es el cuello de botella a dim grande con pocos datos

---

## Experimentos prioritarios

### 🔴 EXP-3: Fix Weight Tying + SpectralThinker Nano v2 (URGENTE)

**Objetivo:** Reentrenar el Nano con weight tying correcto y comparar.

**Cambio:** En `SpectralThinker.__init__`, el orden correcto es:
```python
self.output = nn.Linear(vocab_size, dim, bias=False)
self.tok_embeddings = nn.Embedding(vocab_size, dim)
self.tok_embeddings.weight = self.output.weight  # tying correcto
nn.init.normal_(self.output.weight, std=1/sqrt(dim))
```
Con weight tying, el modelo pasa de 8.64M a **4.44M params totales** (embedding compartido).

**Hipótesis:** Val_loss mejor o igual con la mitad de params en embedding,
porque el modelo no divide gradientes entre dos copias.

**Config:** Igual que train_spectral_nano.yaml, 5000 iters.
**Tiempo estimado:** ~6h en CPU.

---

### 🟡 EXP-4: SpectralThinker Midi (dim=512, k=128)

**Objetivo:** Escalar tanto dim como k proporcionalmente. Balance óptimo entre
capacidad espectral y tamaño de embedding.

**Arquitectura:**
- `dim=512`, `n_layers=8`, `n_heads=8`, `n_kv_heads=4`
- `k_dim_attn=128`, `k_dim_ffn=128`, `k_hidden_ffn=256`
- Embedding: 16384×512 = 8.39M params (con weight tying)
- Núcleos espectrales: 8 × (4×128² + 3×128×256) ≈ 1.31M params
- **Total: ~9.7M params**
- Compresión: 512²/128² = **16x** por proyección

**Ratio datos/params embedding:** 81.9M / 8.39M = 9.8 (mucho mejor que Mega-Nano)

**Hipótesis:** Mejor que el Nano en val_loss final por mayor dim y k.
**Tiempo estimado:** ~20-25h en CPU (5000 iters).

---

### 🟡 EXP-5: Curva k — ¿Cuánto k necesita dim=256?

**Objetivo:** Estudiar la capacidad espectral en función de k, manteniendo dim=256.

| Sub-exp | k | Proj params | Compresión | Hipótesis |
|---------|---|-------------|-----------|-----------|
| 5a | 32 | ~61K | 64x | Aprende? Límite inferior |
| 5b | 64 | 245K | 16x | Nano actual (baseline) |
| 5c | 128 | 983K | 4x | Mejor? |
| 5d | 256 | 3.93M | 1x | Equivale a denso |

Esto permite construir la **curva calidad vs. compresión** del SpectralThinker.
Cada run: 2000 iters, 5 seeds. Tiempo total: ~4 × 2 × 5 × 4s × 2000 = ~44h.

---

### 🟢 EXP-6: Benchmark Velocidad Spectral vs Dense

**Objetivo:** Medir científicamente la ventaja de velocidad/memoria de SpectralThinker
frente a un Transformer denso equivalente.

**Metodología:**
1. Dense (dim=256, 6 capas, full rank) — ~12M params
2. SpectralThinker Nano (dim=256, k=64) — ~4.44M params, misma forward shape
3. Medir: seconds/step, GB RAM, optimizer step time, inference latency

**Resultado esperado:** Dense 1.5-2x más lento por step, 6x más RAM en optimizer.
**Tiempo:** ~30 minutos (no requiere training largo).

---

### 🟢 EXP-7: SpectralThinker con Seq_len=512 (contexto más largo)

**Objetivo:** Probar si la arquitectura espectral mantiene su ventaja con secuencias largas.

Teoría: con seq_len=512 el mecanismo de atención es más costoso (O(S²)),
pero las proyecciones espectrales no cambian — la ventaja relativa debería crecer.

**Config:** Nano config, `seq_len=512`, `batch_size=8`, `grad_accum=8`. 3000 iters.
**Tiempo:** ~12h en CPU.

---

### 🔵 EXP-8: JPEG-LLM — DCT en el KV Cache (Contexto Infinito)

**Objetivo:** Implementar la idea del brainstorming: comprimir el KV cache con DCT.

**Concepto:** En lugar de guardar el KV cache completo (S × D × n_layers floats),
comprimir cada "columna temporal" con DCT y guardar solo los k coeficientes bajos.
- Reducción de memoria: S × D → S × k = S × 64 en lugar de S × 256
- 4x menos memoria de KV cache sin (hipotéticamente) perder información semántica relevante

**Implementación:** Modificar `SpectralAttention` para comprimir/descomprimir K,V en cache.
**Tiempo:** ~1 semana de implementación + 3000 iter de validación.

---

### 🔵 EXP-9: SpectralThinker Multimodal (3D DCT embedding de imágenes)

**Objetivo:** Extender la arquitectura espectral al dominio visual.

**Concepto (de NEW_ALGORITHMS_BRAINSTORMING.md):**
- En lugar de embeddings de tokens de texto, embeddings 2D DCT de patches de imagen
- Una imagen 224×224 = 196 patches de 16×16
- Cada patch se comprime con DCT 2D a k²=64 coeficientes (en lugar de 768 píxeles)
- 12x menos dimensionalidad por patch en el embedding visual

**Conexión:** Unifica el SpectralThinker de texto con visión usando el mismo
formalismo de coeficientes de baja frecuencia.
**Tiempo:** ~2 semanas de implementación.

---

### 🔵 EXP-10: Entrenamiento sin Backprop (Walsh-NEAT Evolutivo)

**Objetivo:** Probar la idea del brainstorming — evolucionar los núcleos DCT/Walsh
en lugar de optimizarlos con gradientes.

**Concepto:** Los núcleos k×k son pequeños (4096 floats para k=64). Una población
de 100 individuos × 4096 floats = 400K floats de genoma total.
Se puede evolucionar con un algoritmo evolutivo (CMA-ES, ES) directamente
sin necesidad de calcular gradientes — relevante para hardware neuromórfico.

**Validación:** Comparar val_loss de ES-SpectralThinker vs Adam-SpectralThinker en MNIST.
**Tiempo:** ~1 semana.

---

## Priorización recomendada

```
Corto plazo (esta semana):
  EXP-3 → fix weight tying, reentrenar Nano v2
  EXP-6 → benchmark velocidad (30 min)

Medio plazo (próximas 2 semanas):
  EXP-4 → Midi (dim=512, k=128)
  EXP-5 → Curva k (validación de capacidad espectral)

Largo plazo (investigación avanzada):
  EXP-7 → Contexto largo
  EXP-8 → JPEG-LLM KV cache
  EXP-9 → Multimodal
  EXP-10 → Walsh-NEAT evolutivo
```

---

*"The question is not how many parameters you have. It is how many frequencies you listen to."*
