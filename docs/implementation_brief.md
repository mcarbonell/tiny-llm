# Implementation Brief — Entrega a otro modelo

**Contexto:** Este documento resume las tareas priorizadas identificadas en `project_review.md` y `strategy_analysis.md`. Está diseñado para ser ejecutado por un agente de código de forma autónoma.

**Reglas del proyecto:** Leer obligatoriamente `GEMINI.md` en la raíz del repo antes de hacer cualquier cambio. Define estándares de logging, nombrado, versionado y restricciones de ejecución.

**Restricción crítica:** NO lanzar entrenamientos ni finetunes. Solo proporcionar los comandos al usuario. Hay un entrenamiento activo en la GPU local.

---

## BLOQUE 1: Bugfixes (Prioridad ALTA)

### 1.1 — train.py: Constantes hardcodeadas ignoran el YAML

**Archivo:** `scripts/train.py`

**Problema:** Las funciones `get_lr()` y `estimate_loss()` usan constantes `DEFAULT_*` (líneas 22-30) en lugar de los valores cargados desde el config YAML. El warmup real es 200 cuando el YAML dice 1000.

**Fix:** Hacer que `get_lr()` lea `warmup_iters` y `min_lr` desde `args_cli`, y `estimate_loss()` lea `eval_iters` desde `args_cli`. Mantener los `DEFAULT_*` solo como fallback si el YAML no los define.

Las variables afectadas son:
- `get_lr()` línea ~279: usa `DEFAULT_WARMUP` (200) → debe usar `getattr(args_cli, 'warmup_iters', 200)`
- `get_lr()` línea ~282-285: usa `DEFAULT_MIN_LR` (1e-5) → debe usar `getattr(args_cli, 'min_lr', 1e-5)`
- `estimate_loss()` línea ~267-268: usa `DEFAULT_EVAL_ITERS` (20) → debe usar `getattr(args_cli, 'eval_iters', 20)`

**Nota:** `get_lr` y `estimate_loss` son funciones internas de `main()` que tienen acceso a `args_cli` por closure. No hay que pasar args extra.

### 1.2 — eval.py: Tokenizador viejo hardcodeado

**Archivo:** `scripts/eval.py`

**Problema:** Línea 50 carga `model/tokenizer.json` (viejo, vocab diferente) en vez de `model/tokenizer_v1.json`. Esto desincroniza la evaluación con el modelo V1.

**Fix:** Cambiar la ruta a `model/tokenizer_v1.json`, o mejor aún, aceptar `--tokenizer_path` como argumento CLI con default `model/tokenizer_v1.json`.

### 1.3 — train.py: tokenizer_path default incorrecto

**Archivo:** `scripts/train.py`, línea 48

**Problema:** `default='model/tokenizer.json'` apunta al tokenizador viejo.

**Fix:** Cambiar default a `'model/tokenizer_v1.json'`.

---

## BLOQUE 2: Flash Attention — Ajuste is_causal (Prioridad MEDIA)

**Archivo:** `model/model.py`, clase `Attention`, método `forward`

**Problema:** Se construye una máscara causal O(n²) manualmente y se pasa con `is_causal=False`. Esto impide que PyTorch use el kernel FlashAttention-2 optimizado en CUDA.

**Fix:** Cuando no hay KV-cache (`past_key_value is None`) y `seqlen > 1`, usar `is_causal=True` sin máscara:
```python
if past_key_value is None and seqlen > 1:
    output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=0.0, is_causal=True)
else:
    output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask, dropout_p=0.0, is_causal=False)
```

**Requisito:** También hay que eliminar la construcción de `mask` en el forward de `TinyThinker` cuando no se necesite (optimización, no bloqueante). O simplemente seguirla construyendo y no pasarla — el código ya tiene `mask = None` como caso base.

**IMPORTANTE:** Esto NO afecta a DirectML (sigue usando la implementación naive). Solo beneficia a CUDA. Asegurarse de que los tests (`tests/test_model.py`) sigan pasando.

---

## BLOQUE 3: Configs de modelos pequeños (Prioridad MEDIA)

Crear 3 nuevos archivos YAML en `configs/`:

### `configs/train_v1_nano_10M.yaml`
```yaml
# TinyThinker 10M (Nano) — Experimentación rápida
dim: 192
n_layers: 6
n_heads: 6
n_kv_heads: 3
vocab_size: 16384
max_seq_len: 512

batch_size: 16
seq_len: 512
max_iters: 5000
learning_rate: 0.0003
min_lr: 0.00001
warmup_iters: 500
eval_interval: 250
eval_iters: 20
weight_decay: 0.1
grad_clip: 1.0
grad_accum_steps: 2
use_gradient_checkpointing: false

data_path: data/train_v1.bin
tokenizer_path: model/tokenizer_v1.json
checkpoint_dir: checkpoints
arch: dense
device: dml
```

### `configs/train_v1_micro_20M.yaml`
```yaml
# TinyThinker 20M (Micro)
dim: 256
n_layers: 8
n_heads: 8
n_kv_heads: 4
vocab_size: 16384
max_seq_len: 512

batch_size: 8
seq_len: 512
max_iters: 5000
learning_rate: 0.0003
min_lr: 0.00001
warmup_iters: 500
eval_interval: 250
eval_iters: 20
weight_decay: 0.1
grad_clip: 1.0
grad_accum_steps: 4
use_gradient_checkpointing: false

data_path: data/train_v1.bin
tokenizer_path: model/tokenizer_v1.json
checkpoint_dir: checkpoints
arch: dense
device: dml
```

### `configs/train_v1_mini_30M.yaml`
```yaml
# TinyThinker 30M (Mini)
dim: 320
n_layers: 10
n_heads: 8
n_kv_heads: 4
vocab_size: 16384
max_seq_len: 512

batch_size: 8
seq_len: 512
max_iters: 5000
learning_rate: 0.0002
min_lr: 0.00001
warmup_iters: 500
eval_interval: 250
eval_iters: 20
weight_decay: 0.1
grad_clip: 1.0
grad_accum_steps: 4
use_gradient_checkpointing: false

data_path: data/train_v1.bin
tokenizer_path: model/tokenizer_v1.json
checkpoint_dir: checkpoints
arch: dense
device: dml
```

**Nota sobre LR:** Se usa 3e-4 para los modelos pequeños (vs 1e-4 del 78M). Los modelos más pequeños toleran y necesitan LRs más altas. Esto será validado en los sweeps.

**Nota sobre `n_heads` y `dim`:** En la config Nano (10M), dim=192 y n_heads=6 da head_dim=32. Es más pequeño que los 64 del modelo 78M, pero funcional. Si da problemas con RoPE, se puede subir a dim=256 con n_layers=4 para mantener ~10M params.

---

## BLOQUE 4: Actualización de modal_train.py (Prioridad MEDIA)

**Archivo:** `scripts/modal_train.py`

**Problemas actuales:**
1. Apunta a `train_combined.bin` (legacy) → cambiar a `train_v1.bin`
2. No usa `--config` → debe aceptar config YAML
3. Argumentos hardcodeados en líneas 69-74 → parametrizar

**Fix recomendado:** Hacer que la función `train()` acepte parámetros y use el flag `--config`:
```python
# Inyeccion de argumentos para train.py
test_args = [
    "scripts/train.py",
    "--config", "/root/configs/train_v1_high_density.yaml",
    "--device", "cuda",
]
```

También actualizar:
- `DATA_VOL_PATH` → `"/vol/data/train_v1.bin"`
- `LOCAL_DATA_FILE` → incluir `train_v1.bin`
- Considerar cambiar GPU a `"l4"` (mejor relación rendimiento/coste para modelos <100M)

---

## BLOQUE 5: Documentación (Prioridad BAJA)

### README.md
- Actualizar badge de params: 12.46M → 78M
- Actualizar descripción general para reflejar el estado V1
- Actualizar rutas en Quick Start (tokenizador, dataset)

### DATASET_V1_README.md
- Completar "Tokens totales": 107,919,573 tokens (ya conocido, falta escribirlo)
- Corregir la descripción de proporciones: PROJECT_STATUS dice 15% lógica, pero el README del dataset dice 10% lógica + 5% planning. Unificar.

### PROJECT_STATUS.md
- Añadir referencia al entrenamiento en curso con link al log

---

## Orden de ejecución recomendado

1. **Bloque 1** (bugfixes) — Impacto inmediato, necesario antes de cualquier nuevo entrenamiento
2. **Bloque 3** (configs) — Crear archivos YAML, no requiere modificar código existente
3. **Bloque 2** (Flash Attention) — Mejora de rendimiento para CUDA, correr tests después
4. **Bloque 4** (Modal) — Actualizar para usar V1
5. **Bloque 5** (docs) — Cuando todo lo anterior esté estable

## Verificación

Después de los bloques 1-3, correr:
```bash
python -m pytest tests/ -v
```
Para verificar que nada se ha roto. Los tests existentes (`test_model.py`, `test_integration.py`, `test_rope.py`) deberían seguir pasando.
