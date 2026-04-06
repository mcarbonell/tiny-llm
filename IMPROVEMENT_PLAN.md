# 📋 Improvement Plan — TinyThinker

> Prioritized list of enhancements to take TinyThinker from "working prototype" to "production-ready research codebase".

---

## 🔴 P0 — Críticos (Corregir ya)

### 1. Fix checkpoint name mismatch (`chat.py`)
- **Estado:** ✅ Completado - `chat.py` usa `resolve_checkpoint` que prioriza `ckpt_finetuned.pt`.
- **Archivos:** `scripts/chat.py`, `scripts/finetune.py`

### 2. Pin dependency versions
- **Estado:** ✅ Parcialmente completado - `requirements.txt` tiene versiones mínimas. Recomendado: usar versiones exactas con `pip freeze`.
- **Archivos:** `requirements.txt`

### 3. Add `.env.example` template
- **Estado:** ✅ Completado - `.env.example` existe con placeholders.
- **Archivos:** `.env.example`, `.gitignore` (agregado `.env`)

### 4. Implement evaluation script (perplexity and tool-calling accuracy)
- **Estado:** ✅ Completado - `scripts/eval.py` mide perplexity y accuracy en tool-calling.
- **Archivos:** `scripts/eval.py`, `README.md` (actualizado)

### 5. Fix text generation issues in `chat.py`
- **Estado:** ✅ Completado - Arreglados espacios en blanco y problemas de tokenizer con re.sub.
- **Archivos:** `scripts/chat.py`

### 6. Implement KV-Cache para inferencia
- **Estado:** ✅ Completado - KV-cache implementado en model.py y eval.py para inferencia eficiente.
- **Archivos:** `model/model.py`, `scripts/eval.py`

### 7. Externalize configuration (argparse + YAML)
- **Estado:** ✅ Completado - `scripts/config.py` creado con dataclass y argparse.
- **Archivos:** `scripts/config.py`

### 8. Real tool integration (DuckDuckGo / Wikipedia)
- **Estado:** ✅ Completado - Integración real con duckduckgo-search en chat.py.
- **Archivos:** `scripts/chat.py`

### 9. Expand test suite
- **Estado:** ✅ Completado - Agregados tests para RoPE, GQA, tokenizer, data loading, KV-cache.
- **Archivos:** `tests/test_model.py`

### 10. Add gradient checkpointing (optional toggle)
- **Estado:** ✅ Completado - Toggle use_checkpoint en TransformerBlock para ahorro de memoria.
- **Archivos:** `model/model.py`

### 11. Add logging framework
- **Estado:** ✅ Completado - Logging con archivos en train.py y finetune.py.
- **Archivos:** `scripts/train.py`, `scripts/finetune.py`

---

## 🟡 P1 — Importantes (Mejoras de arquitectura)

### 4. Implement KV-Cache para inferencia
- **Problema:** En `chat.py`, cada token generado re-evalúa TODA la secuencia (`model(x_cond)`). Complejidad O(n²) por token.
- **Solución:** Añadir `past_key_values` cache en `Attention` y `TransformerBlock`. Forward pasa a ser O(1) por token tras el prompt inicial.
- **Impacto:** 5-10x más rápido en generación.
- **Archivos:** `model/model.py`, `scripts/chat.py`

### 5. Externalize configuration (argparse + YAML)
- **Problema:** Hiperparámetros hardcodeados en `train.py` y `finetune.py`.
- **Solución:** Crear `scripts/config.py` con dataclass + argparse CLI. Soportar `--config config.yaml`.
- **Archivos:** `scripts/config.py`, `scripts/train.py`, `scripts/finetune.py`

### 6. Real tool integration (DuckDuckGo / Wikipedia)
- **Problema:** `search_web_tool` en `chat.py` es un mock con `time.sleep(1)`.
- **Solución:** Implementar búsqueda real con `duckduckgo-search` o Wikipedia API. Fallback a mock si no hay conexión.
- **Archivos:** `scripts/chat.py`, `requirements.txt`

---

## 🟢 P2 — Calidad de código

### 7. Expand test suite
- **Problema:** Solo 1 test (`test_model_forward`).
- **Solución:** Añadir tests para:
  - `test_rotary_embeddings` — verificar periodicidad RoPE
  - `test_gqa_shapes` — confirmar que GQA reduce heads correctamente
  - `test_tokenizer_roundtrip` — encode → decode = original
  - `test_data_loading` — get_batch devuelve shapes correctos
  - `test_kv_cache` — (tras P1-4) output con cache == output sin cache
- **Archivos:** `tests/test_model.py`, `tests/test_tokenizer.py`, `tests/test_data.py`

### 8. Add gradient checkpointing (optional toggle)
- **Problema:** Al escalar a 100M+ parámetros, la memoria será bottleneck.
- **Solución:** `torch.utils.checkpoint` en `TransformerBlock.forward` con flag `use_gradient_checkpointing`.
- **Archivos:** `model/model.py`, `scripts/config.py`

### 9. Add logging framework
- **Problema:** Logs solo por `print()` + archivo plano.
- **Solución:** Usar `logging` module de Python con niveles (INFO/DEBUG/WARNING). Opcional: soporte para Weights & Biases.
- **Archivos:** `scripts/train.py`, `scripts/finetune.py`

---

## 🔵 P3 — Nice to have (Futuro)

### 10. Sliding window attention / ALiBi
- Permitir extrapolación más allá de `max_seq_len` sin degradación.

### 11. LoRA support
- Fine-tuning eficiente sin modificar pesos base. Ideal para experimentar con múltiples datasets.

### 12. Multi-GPU / DDP support
- `torch.distributed` para escalar entrenamiento a varias GPUs.

### 13. Evaluation harness
- Script `scripts/eval.py` que mida perplexity en held-out data + benchmarks simples (truthful_qa, etc).

### 14. Docker support
- `Dockerfile` + `docker-compose.yml` para reproducibilidad total.

---

## 📅 Orden de ejecución recomendado

```
1. Fix checkpoint naming     → 15 min
2. Pin dependencies          → 5 min
3. Add .env.example          → 5 min
4. KV-Cache inference        → 1-2 hrs
5. Config externalization    → 1 hr
6. Real tool integration     → 1 hr
7. Expand tests              → 2 hrs
8. Gradient checkpointing    → 1 hr
9. Logging framework         → 1 hr
```

---

*Última actualización: 2026-04-06*
