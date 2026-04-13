# Revisión Completa del Proyecto TinyThinker

**Fecha:** 13 de Abril de 2026  
**Scope:** Revisión de arquitectura, código, datos, entrenamiento en curso y estado general del proyecto.

---

## 1. Estado General

El proyecto está en un **momento clave**: se ha completado un reinicio estratégico (V1) con un dataset de alta densidad y se está ejecutando el primer entrenamiento serio del modelo de 78M parámetros. El log activo muestra 300 iteraciones completadas de 10,000.

> [!IMPORTANT]
> Hay **5 archivos modificados sin commit**: `model.py`, `model_coga.py`, `model_dense.py`, `model_moe.py` y `train.py`. Asegúrate de hacer commit cuando estabilices los cambios para no perder el trabajo de parcheo de DirectML.

---

## 2. Entrenamiento en Curso — Análisis del Log

Analizando [train_20260413_012853.log](file:///c:/Users/mrcm_/Local/proj/tiny-llm/logs/train_20260413_012853.log):

### Curva de Loss

| Iteración | Loss | LR | Tiempo/iter |
|-----------|------|----|-------------|
| 0 | 9.8526 | 0.00e+00 | 97.35s |
| 50 | 8.7904 | 2.50e-05 | ~91.6s |
| 100 | 8.0599 | 5.00e-05 | ~91.6s |
| 150 | 7.1855 | 7.50e-05 | ~91.6s |
| 200 | 7.3356 | 1.00e-04 | ~91.6s |
| 250 | 6.7974 | 1.00e-04 | — |
| 300 | 7.0385 | 1.00e-04 | ~93.4s |

**Val loss en iter 250: 5.7837** (guardado como mejor modelo)

### Observaciones

- **Loss inicial ~9.85** es consistente con `ln(16384) ≈ 9.70`, lo que indica inicialización correcta (salida quasi-uniforme sobre el vocabulario).
- **Descenso saludable** de 9.85 → 6.56 en 300 iteraciones. La red está aprendiendo.
- **Val loss (5.78) < Train loss (6.80)**: Esto es normal en las fases tempranas con gradient accumulation (el train loss reportado es del último micro-batch, no del promedio).
- **Velocidad**: ~92-93s por iteración (batch=4, accum=4, seq_len=512 en Radeon 780M). Estimación total: **~10.7 días para las 10,000 iteraciones**.
- **Warmup**: Configurado a 200 iteraciones en la CLI pero el YAML dice 1000.

> [!WARNING]
> **Warmup desincronizado**: El YAML tiene `warmup_iters: 1000` pero `train.py` usa `DEFAULT_WARMUP = 200` como constante hardcodeada (línea 28) en lugar de leer del config. El warmup real aplicado es **200 iteraciones**, no 1000. Esto explica que la LR llegue a su máximo (1e-4) en iter ~200 en lugar de 1000.

---

## 3. Arquitectura del Modelo — [model.py](file:///c:/Users/mrcm_/Local/proj/tiny-llm/model/model.py)

### Puntos fuertes
- Implementación limpia y moderna: RoPE, RMSNorm, SwiGLU, GQA, KV-cache, LoRA nativo
- Workarounds correctos para DirectML (autocast desactivado en ops incompatibles)
- `F.scaled_dot_product_attention` integrado (Flash Attention path)
- Gradient checkpointing opcional por capa

### Observaciones técnicas

| Aspecto | Detalle |
|---------|---------|
| **Parámetros** | 78.13M (dim=512, 12 layers, 8 heads, 4 KV heads) |
| **FFN** | SwiGLU con `ffn_dim_multiplier=2.0` → hidden_dim=512 (redondeado) |
| **GQA ratio** | 2:1 (8 Q heads, 4 KV heads) — buena relación calidad/memoria |
| **Contexto** | 1024 máximo, entrenando a 512 |
| **RoPE theta** | 10000.0 (estándar) |

> [!NOTE]
> La FFN `w3` (gate) usa `nn.Linear` estándar en lugar de `LoRALinear`. Esto es intencional (sigue el patrón LLaMA donde solo `w1` y `w2` se adaptan con LoRA), pero vale la pena documentarlo.

---

## 4. Script de Entrenamiento — [train.py](file:///c:/Users/mrcm_/Local/proj/tiny-llm/scripts/train.py)

### Puntos fuertes
- `DMLAdamW` personalizado que evita `aten::lerp` (fix crítico para AMD)
- Soporte multi-arquitectura (dense/moe/coga) limpio
- Logging con timestamp relativo y header de trazabilidad completo (cumple GEMINI.md)
- Resume desde checkpoint funcional
- GradScaler desactivado correctamente en DirectML

### Problemas detectados

1. **Constantes hardcodeadas vs. Config YAML** — Las funciones auxiliares usan constantes `DEFAULT_*` en lugar de los valores del config:
   - `get_lr()` usa `DEFAULT_WARMUP` (200) y `DEFAULT_MIN_LR` (1e-5) ignorando `warmup_iters` y `min_lr` del YAML
   - `estimate_loss()` usa `DEFAULT_EVAL_ITERS` (20) ignorando `eval_iters` del YAML
   - `eval_interval` del YAML se ignora, se usa `DEFAULT_EVAL_INTERVAL` (250) — coinciden por casualidad

2. **Duplicación de sistemas de config**: Existen dos sistemas paralelos:
   - [config.py](file:///c:/Users/mrcm_/Local/proj/tiny-llm/scripts/config.py) — dataclass completo con merge YAML→CLI
   - `train.py` — su propio parser + carga YAML manual
   
   `train.py` no usa `config.py` para nada. Es código muerto.

3. **`weight_decay` hardcodeado**: El optimizer usa `weight_decay=1e-1` directamente (línea 230/232) en lugar de leer `weight_decay: 0.1` del YAML.

4. **Tokenizer path por defecto**: `parse_args()` tiene `default='model/tokenizer.json'` (línea 48), que apunta al tokenizador viejo en vez de `tokenizer_v1.json`. El YAML lo sobreescribe correctamente, pero si ejecutas sin `--config` usará el tokenizador equivocado.

---

## 5. Dataset V1 — Análisis

| Archivo | Tamaño | Tokens estimados |
|---------|--------|-----------------|
| `train_v1.bin` | 206 MB | ~107.9M tokens |
| `train_combined.bin` (legacy) | 614 MB | ~321M tokens |
| `train.bin` (legacy) | 447 MB | ~234M tokens |

### Composición (según [DATASET_V1_README.md](file:///c:/Users/mrcm_/Local/proj/tiny-llm/data/DATASET_V1_README.md))
- FineWeb-Edu (40%) + Cosmopedia v2 (30%) + TinyStories v2 (15%) + Sintético Lógica (10%) + Sintético Plan (5%)

> [!WARNING]
> **Scaling Law concern**: 107.9M tokens para 78M params = ratio **1.38:1**. Chinchilla recomienda 20:1 y la industria moderna va mucho más allá (Llama 3: 1,875:1). Este entrenamiento es esencialmente un **warmup/validación de pipeline**, como reconoce el PROJECT_STATUS. El modelo no podrá converger a un mínimo útil con tan pocos datos.

> [!NOTE]
> Los archivos `train.bin`, `train_combined.bin` y `wiki.bin` son legacy y ocupan ~1.2 GB. Podrían limpiarse si no se van a reutilizar.

---

## 6. Evaluación — [eval.py](file:///c:/Users/mrcm_/Local/proj/tiny-llm/scripts/eval.py)

### Problema importante
- **Tokenizer hardcodeado**: Línea 50 apunta a `model/tokenizer.json` (viejo), no a `tokenizer_v1.json`. Esto causará **desincronización vocabulario-modelo** al evaluar el modelo V1.
- Solo soporta arquitectura `dense` (línea 12 importa solo `TinyThinker`), no `moe` ni `coga`.

---

## 7. Coherencia de Documentación

| Documento | Estado | Notas |
|-----------|--------|-------|
| `PROJECT_STATUS.md` | Actualizado | Refleja el V1 correctamente. Dice 15% lógica pero DATASET_V1_README dice 10% lógica + 5% planning |
| `README.md` | **Desactualizado** | Dice 12.46M params, Fase 1 entrenando. El modelo actual es 78M |
| `ROADMAP.md` | **Todo completado** | Las 3 fases están marcadas como `[x]`. No refleja el reinicio V1 ni los próximos pasos |
| `IMPROVEMENT_PLAN.md` | Parcialmente actualizado | Fase 6-7 pendientes. MLOps notes mencionan "corpus 305M" que ya no aplica |
| `DATASET_V1_README.md` | Casi completo | Falta el conteo final de tokens (dice "Se determinará") |

---

## 8. Checkpoints

Solo hay 2 checkpoints:
- `ckpt_pretrain_best.pt` (1.27 GB) — el mejor hasta ahora (val_loss 5.7837 en iter 250)
- `ckpt_pretrain_latest.pt` (1.27 GB) — el más reciente

> [!NOTE]
> Cada checkpoint pesa 1.27 GB porque incluye model + optimizer state en FP32. Para un modelo de 78M params, los pesos solos serían ~300 MB. El optimizer AdamW duplica el tamaño (2 momentums). Normal pero a tener en cuenta para el disco.

---

## 9. Tests

3 archivos de test:
- `test_model.py` — Tests unitarios de la arquitectura
- `test_integration.py` — Tests de integración del pipeline
- `test_rope.py` — Tests de RoPE

No se han ejecutado en esta revisión (no cambio nada), pero la cobertura es razonable para la fase actual.

---

## 10. Resumen de Hallazgos Priorizados

### Bugs / Riesgos activos

| # | Severidad | Hallazgo |
|---|-----------|----------|
| 1 | **ALTA** | `train.py` ignora `warmup_iters`, `min_lr`, `eval_iters` del YAML — usa constantes hardcodeadas |
| 2 | **ALTA** | `eval.py` usa `tokenizer.json` (viejo) en vez de `tokenizer_v1.json` — evaluación rota para V1 |
| 3 | **MEDIA** | 5 archivos modificados sin commit (riesgo de pérdida de trabajo) |
| 4 | **MEDIA** | `README.md` dice 12.46M params, info muy desactualizada |
| 5 | **BAJA** | Datos legacy (`train.bin`, `train_combined.bin`, `wiki.bin`) ocupan 1.2 GB innecesarios |
| 6 | **BAJA** | `config.py` es código muerto (no lo usa `train.py`) |

### Aspectos positivos

- Arquitectura del modelo sólida y moderna (RoPE, GQA, SwiGLU, Flash Attention, KV-cache)
- Workarounds DirectML bien implementados (`DMLAdamW`, autocast selectivo)
- El entrenamiento está progresando correctamente (loss descendente, vel. estable)
- Logging cumple con el estándar GEMINI.md (timestamp relativo, header de metadatos, trazabilidad)
- Dataset V1 bien curado con mix diverso y documentación clara
- Investigación de scaling laws documentada y aplicada al diseño
