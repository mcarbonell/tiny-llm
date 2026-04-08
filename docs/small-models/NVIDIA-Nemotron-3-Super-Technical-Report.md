# NVIDIA Nemotron 3 Super Technical Report

A powerful 4B parameter LLM optimized for inference.

## Overview

NVIDIA Nemotron 3 Super es un modelo de lenguaje de 4 mil millones de parámetros diseñado para inference eficiente. Forma parte de la familia de modelosNemotron y está optimizado para ejecutarse en GPUs NVIDIA con FP8.

El modelo destaca por:
- **Tamaño compacto**: 4B parámetros
- **Alto rendimiento**: Supera modelos de tamaño similar en benchmarks
- **Optimizado para inference**: Soporte FP8 nativo
- **Licencia comercial**: NVIDIA Open Model License

## Arquitectura

### Configuración Base

| Parámetro | Valor |
|-----------|-------|
| Parámetros totales | 4B |
| Capas | 32 |
| Dimensión del embedding | 3072 |
| Cabezas de atención | 24 |
| Dimensión por cabeza | 128 |
| Vocabulario | ~200,000 tokens |

### Componentes Clave

**Rotary Position Embedding (RoPE)**
- Embedding posicional rotativo para capturar dependencias de distancia
- Implementación optimizada para inference

**SwiGLU**
- Activación gated linear units con función Swish
- Proporciona no-linealidad adaptativa

**QKNorm**
- Normalización de Query y Key antes del attention
- Estabiliza el training y mejora la convergencia

### Arquitectura del Transformer

- **Tipo**: Decoder causal-only
- **Attention**: Multi-head self-attention con RoPE
- **FFN**: SwiGLU activation
- **Normalization**: QKNorm + RMSNorm
- **Dropout**: No utilizado durante inference

## Training

### Dataset

- **Tokens totales**: 8.7T tokens
- **Idiomas**: Más de 12 idiomas
- **Fuentes**: Mix de datasets diversificados

### Proceso de Training

1. **Pre-training**: Training inicial en corpus grande
2. **Post-training**: Fine-tuning con preference data
3. **Quality filtering**: Filtering de baja calidad

### Hardware

- Entrenado en clusters de GPUs NVIDIA
- Optimizado para FP8 training

## Evaluación

### Benchmarks

| Benchmark | Nemotron 3 Super | Mistral-Small-3.1 | Phi-4 | Qwen-2.5-3B |
|-----------|------------------|-------------------|-------|------------|
| MT-Bench | 7.54 | 7.40 | 7.31 | 7.01 |
| HumanEval | 62.20% | 56.10% | 62.80% | 53.70% |
| Math | 55.10% | 52.60% | 42.50% | 51.20% |

### Resultados Clave

- **MT-Bench**: 7.54 (mejor en su categoría)
- **HumanEval**: 62.20% (code generation competitivo)
- **Math**: 55.10% (excelente en tareas matemáticas)

El modelo supera consistentemente a Mistral-Small-3.1, Phi-4 y Qwen-2.5-3B en las mayoría de métricas.

## Inference

### Optimizaciones FP8

El modelo está optimizado para inference en FP8:
- **Precisión primaria**: FP8
- **Fallback**: FP16 si no hay soporte FP8

###Deployment

Compatible con:
- NVIDIA TensorRT-LLM
- vLLM
- Hugging Face Transformers

### Requisitos de Hardware

- **Mínimo**: GPU con soporte FP8 (Ada Lovelace+)
- **VRAM**: ~8GB en FP8, ~8GB adicionales en KV cache

## Uso

### Chatbot / Instruct

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "nvidia/Nemotron-3-Super-4B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="float8_e4m3fn")
```

### FP8 Inference

```python
# Inference con FP8
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Licencia

**NVIDIA Open Model License**

- Uso comercial permitido
- Redistribución permitida
- Modificaciones permitidas
- No requiere atribución específica

Consultar el archivo de licencia para detalles completos.

## Referencias

- [NVIDIA Nemotron 3 Super Hugging Face](https://huggingface.co/nvidia/Nemotron-3-Super-4B-Instruct)
- [NVIDIA TensorRT-LLM](https://developer.nvidia.com/tensorrt-llm)
- [FP8 Specification](https://docs.nvidia.com/cuda/parallel-thread-execution/)

---

*Documento generado desde NVIDIA-Nemotron-3-Super-Technical-Report.pdf*