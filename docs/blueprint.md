# blueprint.md
# Proyecto: Tiny Tool-Using LLM From Scratch
# Objetivo: aprender, experimentar y construir un mini-LLM desde cero con razonamiento corto, uso básico de herramientas y baja dependencia de memoria factual.

---

## 1. Visión del proyecto

Este proyecto busca construir **desde cero** un modelo de lenguaje pequeño, moderno y experimental, con estas propiedades:

- entrenado desde cero, no basado en un modelo industrial preexistente
- orientado a **aprendizaje e investigación personal**
- tamaño lo bastante pequeño para ser viable con recursos modestos o alquiler ocasional de GPU
- capaz de:
  - generar texto coherente
  - seguir instrucciones básicas
  - hacer razonamiento corto/moderado
  - decidir cuándo necesita información externa
  - usar una herramienta simple de búsqueda web
  - responder con cautela cuando no sabe algo

No se busca:
- competir con LLMs frontier
- maximizar conocimiento memorizado
- optimizar para programación
- construir un agente complejo con muchas herramientas desde el principio

La filosofía del proyecto es:

> entrenar una política de comportamiento lingüístico y de consulta externa, no una enciclopedia.

---

## 2. Objetivos concretos

### 2.1 Objetivos principales
1. Implementar un **decoder-only transformer** desde cero.
2. Entrenarlo sobre un corpus limpio y relativamente pequeño.
3. Añadir una fase de instruction tuning.
4. Añadir una fase de entrenamiento para uso de herramientas.
5. Integrarlo con una herramienta simple:
   - `search_web(query)`
6. Conseguir que el modelo:
   - responda con coherencia
   - siga instrucciones
   - haga razonamiento corto
   - prefiera buscar cuando la información es temporal, específica o incierta

### 2.2 Objetivos secundarios
- comparar técnicas clásicas y modernas:
  - LayerNorm vs RMSNorm
  - GELU vs SwiGLU
  - learned positional embeddings vs RoPE
  - datasets limpios vs más ruidosos
  - pretraining-only vs +instruction tuning vs +tool tuning
- diseñar un pipeline reproducible
- aprender sobre:
  - tokenización
  - limpieza de datos
  - entrenamiento autoregresivo
  - evaluación
  - serving e inferencia local

### 2.3 No-objetivos
- RLHF completo
- entrenamiento multimodal
- razonamiento matemático avanzado
- entrenamiento a escala >1B params
- crawling masivo sin filtrar
- tool ecosystem complejo desde fase 1

---

## 3. Alcance inicial

### MVP funcional esperado
Un sistema compuesto por:
1. un mini modelo autoregresivo entrenado desde cero
2. un tokenizer propio
3. un corpus base para pretraining
4. un corpus de instrucciones
5. un corpus de uso de herramientas
6. un bucle de inferencia con:
   - generación
   - parseo de tool calls
   - ejecución de `search_web`
   - reinyección del resultado
   - respuesta final del modelo

### Capacidades mínimas esperadas del MVP
- completar texto
- responder preguntas simples
- resumir
- pedir aclaración en algunos casos
- decidir buscar para preguntas temporales o específicas
- leer el resultado de búsqueda y responder con él

### Capacidades no exigidas al MVP
- planificación larga
- robustez alta en tareas de varios pasos
- browsing complejo multi-hop
- recuperación documental sofisticada
- herramientas múltiples con estado

---

## 4. Idioma

### Recomendación inicial
**Inglés**

Razones:
- mayor disponibilidad de datasets
- mejores recursos de instruction tuning
- más material abierto de alta calidad
- más ejemplos de tool-use
- más benchmarks

### Extensión futura
- variante en español
- pequeño corpus bilingüe
- adaptación/fine-tuning de comportamiento en español

---

## 5. Arquitectura del modelo

## 5.1 Tipo
- decoder-only transformer causal

## 5.2 Configuración recomendada inicial
Dos escalas sugeridas:

### Escala A: didáctica
- parámetros: 30M–60M
- capas: 8–12
- dimensión modelo: 384–512
- cabezas: 6–8
- contexto: 512
- ideal para validar pipeline y experimentar rápido

### Escala B: principal
- parámetros: 80M–150M
- capas: 12–16
- dimensión modelo: 512–768
- cabezas: 8–12
- contexto: 1024
- mejor equilibrio entre aprendizaje y comportamiento útil

### Recomendación principal
Empezar por una configuración pequeña tipo:
- `n_layers = 12`
- `d_model = 512`
- `n_heads = 8`
- `ffn_mult ≈ 4` o variante SwiGLU equivalente
- `context_length = 1024`
- `vocab_size ≈ 16k–32k`

Y después escalar según resultados.

## 5.3 Componentes modernos recomendados
- embeddings token
- RoPE para posiciones
- RMSNorm
- MLP con SwiGLU
- causal self-attention
- weight tying entre embedding y lm_head
- AdamW
- gradient clipping
- cosine decay + warmup

## 5.4 Baselines a comparar
- LayerNorm vs RMSNorm
- GELU MLP vs SwiGLU
- learned positions vs RoPE
- context length 512 vs 1024

---

## 6. Tokenización

## 6.1 Estrategia recomendada
- BPE o unigram
- vocabulario de 16k–32k tokens
- entrenado en el mismo idioma principal del corpus

## 6.2 Decisión inicial
**BPE simple** por claridad e implementación más directa.

## 6.3 Requisitos
- tokens especiales:
  - `<bos>`
  - `<eos>`
  - `<pad>` si fuera necesario
  - `<unk>` si el tokenizer lo requiere
  - tokens para herramientas, por ejemplo:
    - `<tool_call>`
    - `</tool_call>`
    - `<tool_result>`
    - `</tool_result>`

## 6.4 Experimentos futuros
- byte-level BPE
- sentencepiece unigram
- análisis de compresión inglés vs español

---

## 7. Diseño del dataset

No se usará un único dataset monolítico. Se diseñará una **mezcla por capas**.

## 7.1 Capas de datos
1. **Pretraining corpus**
2. **Instruction corpus**
3. **Tool-use corpus**
4. **Synthetic reasoning corpus**

---

## 8. Pretraining corpus

## 8.1 Objetivo
Enseñar:
- lenguaje
- continuidad textual
- coherencia
- conocimiento general limitado
- estilo expositivo y procedural básico

## 8.2 Filosofía
Priorizar:
- datos limpios
- datos trazables
- mezcla moderada
- menos cantidad, más control

Evitar:
- scraping web indiscriminado
- ruido excesivo
- duplicación alta
- contenido sin licencia clara si se planea redistribuir

## 8.3 Fuentes sugeridas
- Wikipedia
- documentación técnica abierta
- libros y ensayos con licencia compatible
- artículos educativos abiertos
- how-to / procedural text abierto
- preguntas y respuestas abiertas con licencia clara

## 8.4 Mezcla orientativa
Ejemplo de mezcla inicial:
- 40% Wikipedia / enciclopédico
- 20% documentación técnica
- 15% libros/ensayo limpio
- 15% texto explicativo/QA
- 10% procedural/how-to

## 8.5 Tamaño orientativo
- mínimo funcional: 100M–300M tokens
- recomendable: 300M–1B tokens
- ambicioso: 1B–2B tokens

Para el proyecto inicial:
- apuntar a **300M–800M tokens limpios** si el coste lo permite

---

## 9. Instruction corpus

## 9.1 Objetivo
Enseñar comportamiento:
- seguir instrucciones
- responder de forma estructurada
- resumir
- comparar
- pedir aclaración
- indicar incertidumbre
- abstenerse cuando falte información

## 9.2 Tipos de ejemplos
- summary
- explain simply
- compare A vs B
- classify
- extract fields
- rewrite
- ask follow-up question
- answer cautiously
- state missing information
- say when a search is needed

## 9.3 Fuentes posibles
- datasets abiertos de instruction tuning con licencia compatible
- ejemplos creados manualmente
- ejemplos sintéticos generados por plantillas
- ejemplos generados por un modelo profesor y revisados

## 9.4 Tamaño orientativo
- mínimo: 5k–10k ejemplos
- recomendable: 20k–100k ejemplos

---

## 10. Tool-use corpus

## 10.1 Objetivo
Entrenar al modelo para:
- decidir si necesita una búsqueda externa
- emitir una llamada de herramienta
- leer un resultado
- responder basándose en el resultado
- manejar incertidumbre

## 10.2 Herramienta inicial
Solo una:
- `search_web(query)`

## 10.3 Principio de diseño
La herramienta debe ser simple y transparente.
El objetivo inicial no es crear un agente complejo, sino enseñar el patrón:

1. recibir consulta
2. decidir si necesita búsqueda
3. emitir query
4. leer resultado
5. responder

## 10.4 Formato sugerido de ejemplo
Formato chat con tool call explícito:

```json
{
  "messages": [
    {"role": "user", "content": "What is the current president of Argentina?"},
    {"role": "assistant", "content": "<tool_call>search_web(\"current president of Argentina official source\")</tool_call>"},
    {"role": "tool", "content": "Source: ..."},
    {"role": "assistant", "content": "According to the retrieved source, ..."}
  ]
}