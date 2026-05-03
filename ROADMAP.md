# 🗺️ Roadmap: TinyThinker (LLM Lógico Minimalista)

Este documento centraliza el plan de ataque del proyecto, basado en los blueprints iniciales. Permite llevar un track de nuestro progreso.

## Fase 1: Adquisición del Lenguaje (Pre-training)
- [x] **1.1. Setup Inicial:**
  - [x] Crear estructura de directorios (`model`, `data`, `scripts`, `tests`).
  - [x] Definir `requirements.txt` (PyTorch, transformers, tokenizers, datasets).
- [x] **1.2. Tokenizador y Datos:**
  - [x] Descargar un subtipo de `roneneldan/TinyStories`.
  - [x] Entrenar Tokenizador BPE personalizado (~16k-32k vocabulario).
  - [x] Tokenizar y guardar el dataset de entrenamiento en binario para carga rápida.
- [x] **1.3. Arquitectura del Modelo (`model.py`):**
  - [x] Implementar Decoder-only Transformer genérico.
  - [x] Integrar RoPE (Rotary Position Embeddings).
  - [x] Integrar RMSNorm.
  - [x] Integrar SwiGLU.
  - [x] Integrar GQA (Grouped-Query Attention).
  - [x] Escribir tests unitarios (`tests/test_model.py`) para comprobar dimensiones y pesos.
- [x] **1.4. Bucle de Entrenamiento (`train.py`):**
  - [x] Implementar loop limpio en PyTorch con AdamW + Cosine Decay.
  - [x] Añadir Mixed Precision (AMP - fp16/bf16).
  - [x] Añadir Gradient Accumulation.
  - [x] Ejecutar prueba en entorno local verificando que el *loss* disminuye.

## Fase 2: Razonamiento (Chain of Thought - CoT)
- [x] Descargar y curar un subconjunto lógico/matemático (ej. de `OpenOrca` o `gsm8k`).
- [x] Preparar prompt templates para enseñar pensar "paso a paso" con marcas `<THINK> ... </THINK>`.
- [x] Fine-tuning (Continual Pre-training) del modelo de Fase 1 con el dataset CoT.

## Fase 3: Uso de Herramientas (Tool-Calling)
- [x] **3.1. Generación de Dataset Sintético:**
  - [x] Crear script para generar ejemplos de consulta -> decisión de tool-call -> resultado -> respuesta.
- [x] **3.2. Formato y Entrenamiento:**
  - [x] Implementar tokens especiales (`<TOOL_CALL>`, `</TOOL_CALL>`, `<TOOL_RESULT>`).
  - [x] Entrenar el modelo sobre la sintaxis estricta.
- [x] **3.3. Inferencia Interactiva (`chat.py`):**
  - [x] Desarrollar sistema interactivo que pause la generación de texto al detectar un `<TOOL_CALL>`.
  - [x] Conectar una herramienta real (ej. DuckDuckGo o Wikipedia).
  - [x] Retroalimentar el `<TOOL_RESULT>` al modelo y continuar hasta la respuesta final.

## Fase 4+: Hacia la Arquitectura Cognitiva (TCA)
Esta fase representa la frontera actual del proyecto, centrada en convertir el LLM en un motor de razonamiento autónomo y modular.

### 📄 Documentación de Investigación (Mayo 2026)
- [**Research Proposal: TinyThinker Cognitive Architecture**](docs/research/tinythinker_next/research_proposal.md): Propuesta integral de OS Cognitivo con recurrencia y scratchpad mutable.
- [**Modular Virgin Experts (MVE)**](docs/research/tinythinker_next/modular_virgin_experts_paper.md): Técnica para aprendizaje continuo sin olvido catastrófico mediante slots de expertos reservados.
- [**The Silent Architect: Philosophical Implications**](docs/research/tinythinker_next/scratchpad_philosophical_implications.md): Ensayo sobre la transición del LLM de Sistema 1 a Sistema 2 mediante pensamiento privado.

### 🧪 Objetivos Técnicos
- [x] **Estabilización de Spectral V7:** Superado el cuello de botella del vocabulario mediante cabezal factorizado.
- [ ] **Implementar TCA v2:** Finalizar el prototipo `model_coga_spectral_v2.py` con integración total de herramientas.
- [ ] **Curriculum de Autocorrección:** Entrenar el modelo para detectar y borrar errores en su propio scratchpad.
- [ ] **Inferencia Adaptativa:** Perfeccionar el cerebelo espectral para optimizar el budget de tokens internos por tarea.
