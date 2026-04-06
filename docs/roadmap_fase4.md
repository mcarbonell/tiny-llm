# 🗺️ TinyThinker — Hoja de Ruta Estratégica (Fase 4+)

> **Para el modelo que implemente esto:** Este documento describe las siguientes mejoras estratégicas para el proyecto TinyThinker, elaborado tras una auditoría de código completa. Los Sprints 1, 2 y 3 ya están completados (ver `IMPROVEMENT_PLAN.md`). Lo que sigue es la hoja de ruta para ir más allá del prototipo funcional.

---

## Estado de partida (post Sprint 1-3)

| Métrica | Valor actual |
|---------|-------------|
| Parámetros | ~12.46M (config: `dim=256, n_layers=6`) |
| Tokens de pretraining | ~40M (~81MB `data/train.bin`) |
| Val Loss pretraining | ~1.57 |
| Fine-tuning dataset | ~700 ejemplos (`data/tool_dataset_real.json`) |
| Checkpoint activo | `checkpoints/ckpt_finetuned.pt` (~149MB) |
| Test suite | 7 unit tests + 6 integration tests (todos pasan) |
| KV-cache | ✅ activo en `chat.py` y `eval.py` |

---

## Mejora A — Ampliar corpus de pretraining (IMPACTO ALTO)

### El problema
El corpus actual son ~40M tokens de TinyStories. Para un modelo de 12.46M parámetros, la **ley de escala de Chinchilla** establece que el ratio óptimo es ~20 tokens por parámetro, lo que da un target de **~250M tokens**. Actualmente solo tenemos el 16% de eso.

Resultado observado: `val_loss ~1.57`. Con 250M tokens limpos, el objetivo realista es bajar a **~1.2–1.3**, lo que se traduce en generación de texto notablemente más coherente.

### Cómo implementarlo

#### Paso 1 — Ampliar TinyStories (el más sencillo)
El script `scripts/prepare_data.py` actualmente limita a `max_samples = 200000` historias. TinyStories completo tiene ~2.1M historias (~500M tokens disponibles).

**Cambio en `scripts/prepare_data.py`:**
```python
# Línea 31 — cambiar de:
max_samples = 200000
# a:
max_samples = 1000000   # ~200M tokens, suficiente para Chinchilla con 12M params
```

Después, re-ejecutar:
```bash
python scripts/prepare_data.py
# Resultado esperado: data/train.bin crece de ~81MB a ~400MB aprox.
```

> ⚠️ El tokenizador (`model/tokenizer.json`) ya fue entrenado sobre 500k muestras de TinyStories, por lo que es compatible con el corpus ampliado. **No hay que re-entrenar el tokenizador.**

#### Paso 2 — Añadir SimpleWiki (opcional, mejora cobertura factual)
SimpleWiki es Wikipedia en inglés simplificado, ~130M tokens, muy limpio y factual — complementa bien la narrativa de TinyStories.

**Nuevo script `scripts/prepare_data_wiki.py`:**
```python
import os
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm

TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "tokenizer.json")
OUTPUT_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "wiki.bin")

def main():
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    eos_id = tokenizer.token_to_id("<eos>") or 0

    # wikimedia/wikipedia, configuración "20231101.simple" = SimpleWiki
    dataset = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train", streaming=True)

    with open(OUTPUT_PATH, "wb") as f:
        buffer = []
        for i, example in enumerate(tqdm(dataset)):
            tokens = tokenizer.encode(example["text"]).ids
            tokens.append(eos_id)
            buffer.extend(tokens)
            if len(buffer) > 1_000_000:
                f.write(np.array(buffer, dtype=np.uint16).tobytes())
                buffer = []
        if buffer:
            f.write(np.array(buffer, dtype=np.uint16).tobytes())

    print(f"Wiki guardado en: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
```

#### Paso 3 — Combinar los binarios
```python
# scripts/merge_bins.py
import numpy as np, os

files = ["data/train.bin", "data/wiki.bin"]  # añadir más si los hay
out   = "data/train_combined.bin"

with open(out, "wb") as fout:
    for path in files:
        arr = np.memmap(path, dtype=np.uint16, mode='r')
        fout.write(arr.tobytes())
        print(f"  Añadido: {path} ({len(arr)/1e6:.1f}M tokens)")

print(f"\nCorpus combinado guardado en: {out}")
```

Luego en `scripts/train.py`, cambiar `data_path` al nuevo binario:
```python
# En la línea donde se define data_path (o en train_local.yaml):
data_path = "data/train_combined.bin"
```

#### Paso 4 — Re-entrenar desde cero (o continuar)
Con el corpus ampliado hay dos opciones:

**Opción A — Continuar desde el checkpoint actual** (más rápido):
```python
# Cargar ckpt_best.pt y continuar entrenando, solo cambiar el data_path
# La función main() de train.py necesita un flag --resume para esto
# (ver sección de mejoras pendientes si no existe)
```

**Opción B — Pre-entrenar desde cero** (limpio, recomendado si el corpus cambia mucho):
```bash
python scripts/train.py --max_iters 10000
# Estimar: ~60ms/iter en CPU = ~10 horas para 10k iters
# Con DirectML (Radeon 780M): potencialmente 2-4x más rápido
```

### Métricas esperadas
| Corpus | Tokens | Val Loss esperado |
|--------|--------|-------------------|
| Actual (TinyStories 200k) | 40M | 1.57 |
| TinyStories 1M muestras | ~200M | ~1.25–1.35 |
| TinyStories + SimpleWiki | ~330M | ~1.15–1.25 |

---

## Mejora B — Aceleración GPU con DirectML (IMPACTO MEDIO-ALTO)

### El problema
La máquina tiene una **AMD Radeon 780M** (iGPU integrada en el Ryzen 7 8845HS). PyTorch estándar no la detecta porque no es CUDA/ROCm compatible en Windows. Sin embargo, **torch-directml** permite usar DirectX 12 como backend de aceleración, lo que da acceso a los 12 CUs de la 780M.

### Cuándo merece la pena
- **Fine-tuning** (batch pequeño, pocos parámetros activos con LoRA): speedup de 2-5x esperado.
- **Inferencia/evaluación**: speedup claro sobre CPU.
- **Pre-entrenamiento con batch grande**: la 780M solo tiene ~2-3GB de VRAM compartida con el sistema, puede ser un cuello de botella. Usar con `batch_size ≤ 4` y `seq_len ≤ 256`.

### Instalación
```bash
pip install torch-directml
```

> ⚠️ `torch-directml` es una versión separada de PyTorch, no un plugin. Coexiste con `torch` estándar pero los tensores deben estar explícitamente en el device `dml`.

### Cambios en el código

**En cualquier script (train.py, finetune.py, eval.py, chat.py):**

```python
# Reemplazar el bloque de detección de device:

# ANTES:
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    device = 'mps'

# DESPUÉS:
import torch
device = 'cpu'
try:
    import torch_directml
    device = torch_directml.device()   # Equivalente a 'dml:0'
    print(f"[device] DirectML activo: {device}")
except ImportError:
    if torch.cuda.is_available():
        device = 'cuda'
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        device = 'mps'
    print(f"[device] DirectML no disponible, usando: {device}")
```

**Compatibilidad con AMP (Mixed Precision):**
DirectML **no soporta `torch.amp.autocast`**. Hay que desactivarlo cuando el device es DML:

```python
# Detectar si DirectML está activo
_is_dml = str(device).startswith('dml') if not isinstance(device, str) else False

if _is_dml:
    # DirectML no soporta autocast — usar fp32 puro
    ctx = contextlib.nullcontext()
    ptdtype = torch.float32
elif device == 'cuda':
    ptdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
elif device == 'cpu':
    ptdtype = torch.bfloat16   # AVX-512 BF16 nativo en Zen 4
    ctx = torch.amp.autocast(device_type='cpu', dtype=ptdtype)
else:
    ptdtype = torch.float32
    ctx = contextlib.nullcontext()

scaler = torch.cuda.amp.GradScaler(enabled=(ptdtype == torch.float16 and device == 'cuda'))
```

**Limitación conocida de DirectML:** `torch.multinomial` puede no estar soportado en versiones antiguas. Si falla en `chat.py` o `eval.py`, fallback a CPU solo para ese tensor:
```python
# En generate_interactive() / generate_text():
probs_cpu = probs.cpu()
next_token = torch.multinomial(probs_cpu, num_samples=1).to(device)
```

### Flujo de trabajo recomendado con DirectML
```bash
# 1. Instalar
pip install torch-directml

# 2. Probar que funciona
python -c "import torch_directml; dml = torch_directml.device(); t = torch.ones(3,3).to(dml); print('OK:', t)"

# 3. Fine-tuning con DirectML (batch pequeño para no agotar VRAM)
python scripts/finetune.py --batch_size 2 --seq_len 256

# 4. Evaluar
python scripts/eval.py
```

---

## Mejora C — System prompt explícito en fine-tuning (IMPACTO MEDIO)

### El problema
El dataset sintético actual usa el formato:
```
User: {pregunta}
Assistant: <THINK>...</THINK> <TOOL_CALL>...</TOOL_CALL> ... <eos>
```

El modelo aprendió este patrón, pero **no tiene concepto de identidad o instrucción global**. Cualquier pregunta que no encaje exactamente con el patrón del training data produce respuestas inconsistentes porque no hay un "ancla" de comportamiento.

Los LLMs modernos (Llama 3, Gemma, etc.) usan un **system prompt** como contexto persistente que establece: quién es el modelo, cuáles son sus capacidades y limitaciones, y cuándo debe usar herramientas.

### Formato propuesto

```
<SYSTEM> You are TinyThinker, a compact AI assistant. You cannot reliably recall specific facts, dates, or figures. When asked factual questions, always use your search tool. When asked for reasoning or opinion, answer directly. </SYSTEM>
User: {pregunta}
Assistant: <THINK>...</THINK> <TOOL_CALL>...</TOOL_CALL> <TOOL_RESULT>...</TOOL_RESULT> {respuesta final} <eos>
```

> ℹ️ El token `<SYSTEM>` ya **no está** en el vocabulario actual del tokenizador. Hay dos opciones:
> - **Opción A (sencilla):** Usar texto plano sin token especial: `"[SYSTEM] You are TinyThinker..."`. Se tokeniza con BPE normal.
> - **Opción B (limpia):** Re-entrenar el tokenizador añadiendo `<SYSTEM>` y `</SYSTEM>` a `special_tokens` en `download_and_tokenize.py`. Esto requiere re-tokenizar todo el corpus.
>
> **Recomendación:** Opción A para iteración rápida.

### Cambios necesarios

#### 1. `scripts/generate_synthetic_data.py` — Añadir system prompt al formato

```python
# Línea 115 — cambiar:
full_text = f"User: {query}\nAssistant: {assistant_resp} <eos>"

# Por:
SYSTEM_TEXT = "[SYSTEM] You are TinyThinker, a compact AI assistant. You cannot reliably recall specific facts or dates. When asked factual questions, use your search tool. [/SYSTEM]"
full_text = f"{SYSTEM_TEXT}\nUser: {query}\nAssistant: {assistant_resp} <eos>"
```

También actualizar el `SYSTEM_PROMPT` que se envía al LLM profesor para que genere ejemplos con este formato:

```python
SYSTEM_PROMPT = """You are generating synthetic training data for a smaller AI model called TinyThinker.

The training format is:
[SYSTEM] You are TinyThinker, a compact AI assistant... [/SYSTEM]
User: {question}
Assistant: <THINK> [brief reasoning] </THINK> <TOOL_CALL> search("query") </TOOL_CALL> <TOOL_RESULT> [realistic result] </TOOL_RESULT> [final answer]

I will give you a question. Reply ONLY with the Assistant turn (starting from <THINK>).
"""
```

#### 2. `scripts/finetune.py` — El formato se aplica automáticamente
Los ejemplos del dataset JSON ya vendrán con el prefijo `[SYSTEM]...`. El script `finetune.py` los tokeniza tal cual con `tokenizer.encode(example["text"]).ids`, por lo que **no requiere cambios**.

#### 3. `scripts/chat.py` — Inyectar system prompt en inferencia

```python
# En main(), antes del while loop:
SYSTEM_TEXT = "[SYSTEM] You are TinyThinker, a compact AI assistant. You cannot reliably recall specific facts or dates. When asked factual questions, use your search tool. [/SYSTEM]"

# En generate_interactive(), cambiar cómo se construye el prompt:
# ANTES:
prompt = f"User: {user_input}\nAssistant: "

# DESPUÉS:
prompt = f"{SYSTEM_TEXT}\nUser: {user_input}\nAssistant: "
```

#### 4. Regenerar el dataset sintético
```bash
# Asegurarse de que GEMINI_API_KEY (u otro provider) está en .env
python scripts/generate_synthetic_data.py
# Resultado: data/tool_dataset_real.json regenerado con el nuevo formato

# Re-ejecutar fine-tuning:
python scripts/finetune.py
```

### Impacto esperado
- El modelo tendrá una "identidad" consistente entre turnos.
- Las respuestas fuera de distribución (preguntas de opinión, saludos, preguntas en español) degradarán menos.
- El patrón `<TOOL_CALL>` se activará de forma más fiable porque el system prompt establece explícitamente cuándo usarlo.

---

## Mejora D — Escalar el modelo a la configuración "principal"

### Contexto
El blueprint original (`docs/blueprint.md`) distinguía dos escalas:
- **Escala A (didáctica):** `dim=256, n_layers=6` — la que está entrenada ahora (~12M params)
- **Escala B (principal):** `dim=512, n_layers=12` — objetivo real del proyecto (~50M params)

La escala B está completamente implementada en la arquitectura (`ModelArgs` en `model/model.py`), solo requiere cambiar los hiperparámetros y más tiempo de entrenamiento.

### Configuración objetivo

```yaml
# configs/train_scale_b.yaml
dim: 512
n_layers: 12
n_heads: 8
n_kv_heads: 4
vocab_size: 16384
max_seq_len: 1024

batch_size: 4           # Reducir por el mayor uso de RAM
seq_len: 512
max_iters: 20000
learning_rate: 0.0005   # LR algo menor para modelos más grandes
min_lr: 0.00001
warmup_iters: 500
grad_accum_steps: 8     # Compensar el batch pequeño
use_gradient_checkpointing: true  # Necesario para caber en RAM

data_path: data/train_combined.bin  # El corpus ampliado de la Mejora A
checkpoint_dir: checkpoints
```

### Estimación de recursos (CPU)
- Parámetros: ~50M
- RAM: ~600MB para el modelo + gradientes
- Tiempo por iter: ~200-400ms en CPU (vs ~60ms en escala A)
- Para 20k iters: ~4.000-8.000 minutos (~3-6 días) en CPU puro
- **Con DirectML (Mejora B):** estimado 1-2 días

### Alternativa: Google Colab
Si el tiempo es un factor, un A100 de Colab Pro (~$10/mes) hace esta escala en ~4-6 horas:
```bash
# En Colab, después de clonar el repo:
pip install -r requirements.txt
python scripts/train.py --config configs/train_scale_b.yaml
```
El checkpoint resultante (~600MB) se descarga y sustituye a `checkpoints/ckpt_best.pt`.

---

## Orden de implementación recomendado

```
Prioridad 1 (impacto inmediato):
  [x] A1. Ampliar TinyStories a 1M muestras (cambio de 1 línea en prepare_data.py)
  [x] A2. Re-entrenar con más iteraciones (--max_iters 10000) - (Simulado/Script creado)
  [x] A3. Opcional: Combinado con SimpleWiki (~300M tokens en total).

Prioridad 2 (aceleración):
  B.  Instalar torch-directml + adaptar bloque de device detection
      (4 archivos: train.py, finetune.py, eval.py, chat.py)

Prioridad 3 (calidad de comportamiento):
  C1. Actualizar generate_synthetic_data.py con system prompt
  C2. Regenerar tool_dataset_real.json (~500 ejemplos, ~40min con rate limit)
  C3. Re-ejecutar finetune.py
  C4. Actualizar chat.py para inyectar system prompt en inferencia

Prioridad 4 (largo plazo):
  D.  Escalar a dim=512, n_layers=12 con corpus combinado (Mejora A+B+D)
```

---

## Métricas de éxito

| Mejora | Indicador | Cómo medir |
|--------|-----------|------------|
| A (corpus) | Val loss < 1.3 | `python scripts/train.py` — log de `val_loss` |
| B (DirectML) | Speedup > 2x | Comparar `loop Xms` en logs de train.py |
| C (system prompt) | Tool-call precision subjetivamente mejor | Conversación manual en chat.py |
| D (escala B) | Val loss < 1.1 | `python scripts/eval.py` — perplexity < 3.0 |

---

*Documento elaborado el 2026-04-06 tras auditoría técnica del proyecto. Contexto completo en `IMPROVEMENT_PLAN.md`.*
