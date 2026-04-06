# 📋 Improvement Plan — TinyThinker

> Prioritized list of enhancements to take TinyThinker from "working prototype" to "production-ready research codebase".

---

## 🔴 P0 — Críticos (Corregir ya)

> ⚠️ **Bugs encontrados en auditoría de código (2026-04-06)**

### BUG-A. Double attention call en `TransformerBlock.forward()` [🔴 CRÍTICO]
- **Problema:** Cuando `past_key_value is not None`, `self.attention()` se llamaba **dos veces**: una para obtener la salida `h` y otra para obtener el nuevo KV. Doble cómputo desperdiciado + estados KV inconsistentes.
- **Estado:** ✅ Completado - `attn_out, new_kv` capturados de una sola llamada.
- **Archivos:** `model/model.py` (`TransformerBlock.forward`)

### BUG-B. Dead code inalcanzable en `TinyThinker.forward()` [🔴 CRÍTICO]
- **Problema:** Líneas finales de `forward()` eran código muerto: el `return logits` anterior impeda llegar al bloque duplicado.
- **Estado:** ✅ Completado - Bloque duplicado eliminado.
- **Archivos:** `model/model.py` (`TinyThinker.forward`)

### BUG-C. `out_dir` no definido en scope global de `train.py` [🔴 CRÍTICO]
- **Problema:** `logging.basicConfig()` referenciaba `out_dir` antes de que existiera en ningún scope. Causaba `NameError` al ejecutar el script.
- **Estado:** ✅ Completado - `out_dir` definido en scope global antes del `basicConfig()`.
- **Archivos:** `scripts/train.py`

### BUG-D. KV-cache no usado en `chat.py` [🟡 IMPORTANTE]
- **Problema:** `generate_interactive()` hacía un forward completo sobre toda la secuencia en cada paso (O(n²)).
- **Estado:** ✅ Completado - `generate_interactive()` reescrita con KV-cache: prefijo procesado una vez, luego un token por paso. También añadido logging estructurado a `search_web_tool` y `logging.basicConfig()` en `main()`.
- **Archivos:** `scripts/chat.py`

### BUG-E. Residual connection faltante en gradient checkpointing [🟡 IMPORTANTE]
- **Problema:** Cuando `use_checkpoint=True`, la conexión residual `h = x + h` no se aplicó en la rama de checkpointing.
- **Estado:** ✅ Completado - La residual connection `h = x + attn_out` ahora se aplica en ambas ramas (checkpoint y normal). El código de gradient checkpointing se simplificó eliminando el bloque duplicado.
- **Archivos:** `model/model.py` (`TransformerBlock.forward`)


### 1. Fix checkpoint name mismatch (`chat.py`)
- **Estado:** ✅ Completado - `chat.py` usa `resolve_checkpoint` que prioriza `ckpt_finetuned.pt`.
- **Archivos:** `scripts/chat.py`, `scripts/finetune.py`

### 2. Pin dependency versions
- **Estado:** ⏳ Pendiente - `requirements.txt` usa versiones mínimas. Necesario: versiones exactas con `pip freeze`.
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
- **Estado:** ✅ Completado - Agregadas tests para RoPE, GQA, tokenizer, data loading, KV-cache.
- **Archivos:** `tests/test_model.py`

### 10. Add gradient checkpointing (optional toggle)
- **Estado:** ⏳ Pendiente - Toggle existe pero no hay CLI flag para activarlo.
- **Archivos:** `model/model.py`, `scripts/config.py`, `scripts/train.py`

### 11. Add logging framework
- **Estado:** ✅ Completado - Logging con archivos en train.py y finetune.py.
- **Archivos:** `scripts/train.py`, `scripts/finetune.py`

### 12. Fix code duplication in `model.py`
- **Estado:** ⏳ Pendiente - Cubierto por BUG-A y BUG-B arriba. Los bugs son consecuencia directa de esta duplicación.
- **Archivos:** `model/model.py`

### 13. Add input validation in `eval.py`
- **Estado:** ⏳ Pendiente - No valida estructura de dataset antes de procesar.
- **Archivos:** `scripts/eval.py`

### 14. Add error logging in `chat.py`
- **Estado:** ✅ Completado - Cubierto por BUG-D: logging INFO/WARNING/ERROR en `search_web_tool` y `basicConfig()` en `main()`.
- **Archivos:** `scripts/chat.py`

---

## 🟡 P1 — Importantes (Mejoras de arquitectura)

### 1. YAML config support
- **Estado:** ⏳ Pendiente - `scripts/config.py` soporta argparse pero no YAML.
- **Solución:** Agregar `--config config.yaml` que cargue parámetros desde archivo YAML.
- **Archivos:** `scripts/config.py`, `scripts/train.py`, `scripts/finetune.py`

### 2. Integration tests
- **Estado:** ⏳ Pendiente - Tests unitarios existen, falta end-to-end.
- **Solución:** Agregar tests de integración que validen pipeline completo.
- **Archivos:** `tests/test_integration.py`

---

## 🟢 P2 — Calidad de código

### 1. Expand test suite
- **Estado:** ✅ Completado - Tests ampliados incluyendo RoPE, GQA, tokenizer, data loading, KV-cache y LoRA.
- **Archivos:** `tests/test_model.py`

### 2. Gradient checkpointing CLI flag
- **Estado:** ⏳ Pendiente - Agregar `--use_gradient_checkpointing` a CLI.
- **Archivos:** `scripts/config.py`, `scripts/train.py`

### 3. Logging en tool-calling
- **Estado:** ⏳ Pendiente - Agregar logging a `search_web_tool` con niveles INFO/WARNING/ERROR.
- **Archivos:** `scripts/chat.py`

---

## 🔵 P3 — Nice to have (Futuro)

### 1. Sliding window attention / ALiBi
- Permitir extrapolación más allá de `max_seq_len` sin degradación.

### 2. Multi-GPU / DDP support
- `torch.distributed` para escalar entrenamiento a varias GPUs.

### 3. Docker support
- `Dockerfile` + `docker-compose.yml` para reproducibilidad total.

### 4. Weights & Biases integration
- Soporte opcional para W&B logging en `scripts/train.py`.

---

## 📅 Orden de ejecución recomendado

```
# Sprint 1 — Bugs críticos (< 1h)
1. BUG-A: Fix double attention call (model.py)           → 10 min
2. BUG-B: Remove dead code (model.py)                   → 5 min
3. BUG-C: Fix out_dir scope in train.py                 → 5 min
4. BUG-D: Migrate KV-cache to chat.py                   → 20 min
5. BUG-E: Fix residual connection in grad-ckpt          → 10 min

# Sprint 2 — Mejoras de calidad (< 2h)
6. Add gradient checkpointing CLI flag                   → 15 min
7. Add error logging in search_web_tool (chat.py)        → 15 min
8. Add input validation (eval.py)                        → 20 min
9. Pin dependencies (pip freeze)                         → 5 min
10. YAML config support                                  → 30 min
11. Integration tests                                    → 1 hr
```

---

*Última actualización: 2026-04-06 (auditoría de código — 5 bugs nuevos añadidos)*
