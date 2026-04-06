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
- **Estado:** ✅ Completado - `requirements.txt` actualizado con versiones exactas (`pip freeze`). Añadido `PyYAML==6.0.3` como dependencia explícita.
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
- **Estado:** ✅ Completado - Flag `--use_gradient_checkpointing` añadido a `train.py` (vía argparse) y al dataclass `Config`. Activa `layer.use_checkpoint=True` en todas las capas tras la inicialización del modelo.
- **Archivos:** `model/model.py`, `scripts/config.py`, `scripts/train.py`

### 11. Add logging framework
- **Estado:** ✅ Completado - Logging con archivos en train.py y finetune.py.
- **Archivos:** `scripts/train.py`, `scripts/finetune.py`

### 12. Fix code duplication in `model.py`
- **Estado:** ✅ Completado - Resuelto por BUG-A (doble llamada a `attention()`) y BUG-B (dead code eliminado). El `TransformerBlock.forward()` ahora es lineal y sin duplicaciones.
- **Archivos:** `model/model.py`

### 13. Add input validation in `eval.py`
- **Estado:** ✅ Completado - Función `validate_dataset()` valida tipo lista, no-vacío, y campo `text: str` en los primeros 10 items. Ambas funciones de evaluación llaman a `validate_dataset()` y retornan `None` si el esquema es incorrecto.
- **Archivos:** `scripts/eval.py`

### 14. Add error logging in `chat.py`
- **Estado:** ✅ Completado - Cubierto por BUG-D: logging INFO/WARNING/ERROR en `search_web_tool` y `basicConfig()` en `main()`.
- **Archivos:** `scripts/chat.py`

---

## 🟡 P1 — Importantes (Mejoras de arquitectura)

### 1. YAML config support
- **Estado:** ✅ Completado - `Config.from_args()` acepta `--config ruta.yaml`. Los valores del YAML se cargan primero; los flags CLI los sobreescriben. Método `save_yaml()` para serializar la config actual. Claves desconocidas emiten `Warning`. Añadido `configs/train_local.yaml` como ejemplo documentado.
- **Archivos:** `scripts/config.py`, `configs/train_local.yaml`

### 2. Integration tests
- **Estado:** ⏳ Pendiente - Tests unitarios existen, falta end-to-end.
- **Diseño propuesto:** Ver sección **🧪 Diseño: Integration Tests** más abajo.
- **Archivos:** `tests/test_integration.py`

---

## 🟢 P2 — Calidad de código

### 1. Expand test suite
- **Estado:** ✅ Completado - Tests ampliados incluyendo RoPE, GQA, tokenizer, data loading, KV-cache y LoRA.
- **Archivos:** `tests/test_model.py`

### 2. Gradient checkpointing CLI flag
- **Estado:** ✅ Completado - Cubierto por P0-10. Flag `--use_gradient_checkpointing` activo en `train.py` y `Config`.
- **Archivos:** `scripts/config.py`, `scripts/train.py`

### 3. Logging en tool-calling
- **Estado:** ✅ Completado - Cubierto por BUG-D. Logging INFO/WARNING/ERROR en `search_web_tool`. `logging.basicConfig()` con `chat.log` FileHandler en `main()`.
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

## 📅 Orden de ejecución — Estado

```
# Sprint 1 — Bugs críticos ✅ COMPLETADO (commit bd71c55)
1. BUG-A: Fix double attention call (model.py)           → ✅
2. BUG-B: Remove dead code (model.py)                   → ✅
3. BUG-C: Fix out_dir scope in train.py                 → ✅
4. BUG-D: Migrate KV-cache to chat.py                   → ✅
5. BUG-E: Fix residual connection in grad-ckpt          → ✅

# Sprint 2 — Mejoras de calidad ✅ COMPLETADO (commit 7208076)
6. Add gradient checkpointing CLI flag                   → ✅
7. Add error logging in search_web_tool (chat.py)        → ✅
8. Add input validation (eval.py)                        → ✅
9. Pin dependencies (pip freeze)                         → ✅
10. YAML config support                                  → ✅

# Sprint 3 — Tests de integración ⏳ EN PROGRESO
11. Integration tests                                    → ⏳ (ver diseño abajo)
```

---

## 🧪 Diseño: Integration Tests (`tests/test_integration.py`)

Los tests de integración validan el **pipeline completo** end-to-end sin mocks, usando el modelo y checkpoint reales del proyecto.

### Filosofía
- Cada test valida **un flujo completo**, no un componente aislado.
- Se usan los artefactos reales del proyecto (`ckpt_best.pt`, `tokenizer.json`, `tool_dataset_real.json`).
- Los tests deben ser **rápidos** (se evita entrenamiento desde cero): se usa el checkpoint existente en modo eval.
- Se ejecuta con `pytest -m integration` para poder aislarlos de los unit tests.

### Tests propuestos

| ID | Nombre | Flujo completo cubierto |
|----|--------|-------------------------|
| IT-1 | `test_checkpoint_load_and_generate` | Load checkpoint → tokenize prompt → generate N tokens → assert non-empty output |
| IT-2 | `test_eval_pipeline_perplexity` | Load checkpoint → load dataset → `calculate_perplexity()` → assert is finite float |
| IT-3 | `test_eval_pipeline_tool_accuracy` | Load checkpoint → load dataset → `evaluate_tool_calling_accuracy()` → assert 0 ≤ acc ≤ 1 |
| IT-4 | `test_kv_cache_consistency` | Generate con KV-cache vs. sin KV-cache → assert logits idénticos para misma secuencia |
| IT-5 | `test_yaml_config_roundtrip` | `Config.from_args()` → `save_yaml()` → re-load desde YAML → assert campos iguales |
| IT-6 | `test_dataset_validation_rejects_invalid` | `validate_dataset()` con datasets mal formados → assert raises `ValueError` |

### Decisiones de diseño

- **IT-1/2/3:** Saltar si no existe `checkpoints/ckpt_best.pt` (marker `@pytest.mark.skipif`).
- **IT-4:** Fundamental para garantizar que el KV-cache del BUG-D fix produce resultados correctos.
- **IT-5:** Puro Python, no necesita GPU ni modelo. Siempre ejecuta.
- **IT-6:** Puro Python, siempre ejecuta. Valida los 4 casos de schema inválido.

### Ejecución prevista
```bash
pytest tests/test_integration.py -v                    # todos
pytest tests/ -v                                        # unit + integration
pytest tests/test_integration.py -k "config or valid"  # solo los rápidos
```

---

*Última actualización: 2026-04-06 (Sprint 1 + 2 completados — Diseño IT añadido)*
