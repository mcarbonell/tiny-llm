# Nanbeige4-3B Technical Report

A 3B parameter large language model optimized for Chinese language tasks.

## Overview

Nanbeige4-3B es un modelo de lenguaje de 3 mil millones de parámetros especializado en chino. Desarrollado por Lingyi Software Technology (灵医大模型), destaca en comprensión y generación de lenguaje chino con capacidades de对话(chat) e 指令跟随(instruction following).

El modelo destaca por:
- **Especialización china**: Optimizado para lenguaje chino
- **Tamaño compacto**: 3B parámetros
- **Alto rendimiento**: Supera modelos de tamaño similar en benchmarks chinos
- **Código abierto**: Available on Hugging Face

## Arquitectura

### Configuración Base

| Parámetro | Valor |
|-----------|-------|
| Parámetros totales | 3B |
| Capas | 28 |
| Dimensión del embedding | 3072 |
| Cabezas de atención | 12 |
| Vocabulario | 200,000+ tokens |

### Arquitectura del Transformer

- **Pre-training**: Transformer decoder causal
- **Fine-tuning**: LLM4PK (HuggingFace open source framework)
- **Normalización**: RMSNorm
- **Activación**: SwiGLU / SiLU
- **Positional Encoding**: Rotary Position Embedding (RoPE)

### Componentes Clave

**Pre-training**
- Decoder-only transformer architecture
- RoPE para embeddings posicionales
- SwiGLU como activación

**Fine-tuning (SFT)**
- Framework: LLM4PK (HuggingFace)
- datasets: 1.5Minstruction-response pairs
- Especialización en tareas chino

## Training

### Dataset de Pre-training

- **Tamaño**: 4.5TB corpus
- **Idiomas**: Chino e Inglés
- **Fuentes**: Diversified Chinese and English corpus

### Dataset de Fine-tuning (SFT)

- **Pares instrucciones-respuesta**: 1.5M
- **Idioma**: Principalmente chino
- **Tipos**: Diverse instruction types

### Proceso

1. **Pre-training**: Entrenamiento en corpus grande (4.5TB)
2. **SFT (Supervised Fine-Tuning)**: Fine-tuning con datos instrucción-respuesta
3. **Quality filtering**: Filtrado de calidad

## Evaluación

### Benchmarks

| Benchmark | Nanbeige4-3B | Qwen2-2.5B | MiniMax-M2.2 | Yi-1.5-3B-Chat |
|-----------|--------------|------------|--------------|----------------|
| C-Eval | 74.16 | 72.40 | 68.90 | 71.20 |
| CMMLU | 72.45 | 70.80 | 66.50 | 69.10 |
| AGIEval | 52.70 | 50.20 | 45.80 | 48.90 |
| MMLU | 60.70 | 58.90 | 55.20 | 57.80 |

### Resultados Clave

- **C-Eval**: 74.16 (evaluación complejo chino)
- **CMMLU**: 72.45 (multi-tarea chino)
- **AGIEval**: 52.70 (exámenes universidad china)
- **MMLU**: 60.70 (evaluación multi-idioma)

El modelo supera consistentemente a Qwen2-2.5B-Instruct, MiniMax-M2.2, y Yi-1.5-3B-Chat en benchmarks chinos.

## Uso

### Installation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "LingyiSoftware/Nanbeige4-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
```

### Inference

```python
# Chat completion
messages = [
    {"role": "user", "content": "你好，请介绍一下北京的历史"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=True, temperature=0.7)
outputs = outputs[inputs.input_ids.size(1):]
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### chat-template

El modelo usa un chat template específico para conversaciones:

```
[|im_start|]system
{system_message}
[|im_end|]
[|im_start|]user
{user_message}
[|im_end|]
[|im_start|]assistant
{assistant_message}
[|im_end|]
```

## Limitaciones

- **Especialización**: Optimizado principalmente para chino, rendimiento limitado en otros idiomas
- **Tamaño**: 3B parámetros puede limitar capacidades complejas
- **Conocimiento**: Limitado a datos de training hasta corte temporal

## Ideas para Proyectos Similares

### De este modelo podemos aprender:

1. **Especialización**: Enfoque en un idioma específico puede mejorar rendimiento significativamente
2. **Datos de calidad**: 1.5M pares SFT bien curados son efectivos
3. **Dataset ratio**: 4.5TB pre-training + 1.5M SFT es una proporción razonable
4. **Arquitectura moderna**: RoPE + SwiGLU + RMSNorm es estándar y efectiva
5. **Benchmarking**: Usar benchmarks específicos del dominio (C-Eval, CMMLU para chino)

## Referencias

- [Nanbeige4-3B Hugging Face](https://huggingface.co/LingyiSoftware/Nanbeige4-3B)
- [C-Eval Benchmark](https://cevalbenchmark.com)
- [CMMLU Benchmark](https://github.com/huawei-noah/HEIM)

---

*Documento generado desde Nanbeige4-3B.pdf*