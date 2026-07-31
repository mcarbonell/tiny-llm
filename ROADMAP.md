# 🗺️ Roadmap: TinyThinker (LLM Lógico Minimalista)

> **Filosofía del Viaje:** TinyThinker es la culminación de una espiral evolutiva que comenzó con **SOMA** (gestión soberana de contexto), se estructuró en **COGA** (arquitectura de sistema operativo cognitivo) y se refinó atómicamente con **Attention Neuron** (síntesis espectral y memoria holográfica). Este Roadmap traza el camino hacia un Agente Soberano Total.

## Fase 0: El Legado Genético
- [x] **SOMA (Macro):** Validación de la gestión de contexto soberana y "The Bitter Lesson".
- [x] **COGA (Sistema):** Implementación del Scratchpad y bucle recurrente inicial.
- [x] **Attention Neuron (Átomo):** Descubrimiento de núcleos Matrix-Free, el Hipocampo Holográfico y la Regla Delta en Fase Compleja.

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
  - [x] Integrar RoPE (Rotary Position Embeddings), RMSNorm, SwiGLU, GQA.
  - [x] Escribir tests unitarios (`tests/test_model.py`) para comprobar dimensiones y pesos.
- [x] **1.4. Bucle de Entrenamiento (`train.py`):**
  - [x] Implementar loop limpio en PyTorch con AdamW + Cosine Decay.
  - [x] Añadir Mixed Precision (AMP - fp16/bf16) y Gradient Accumulation.

## Fase 2: Razonamiento (Chain of Thought - CoT)
- [x] Descargar y curar un subconjunto lógico/matemático.
- [x] Preparar prompt templates para enseñar a pensar "paso a paso" con marcas `<THINK> ... </THINK>`.
- [x] Fine-tuning (Continual Pre-training) del modelo de Fase 1 con el dataset CoT.

## Fase 3: Uso de Herramientas (Tool-Calling)
- [x] **3.1. Generación de Dataset Sintético:** Ejemplos de consulta -> decisión de tool-call -> resultado -> respuesta.
- [x] **3.2. Formato y Entrenamiento:** Tokens especiales (`<TOOL_CALL>`, `</TOOL_CALL>`, `<TOOL_RESULT>`).
- [x] **3.3. Inferencia Interactiva (`chat.py`):** Sistema interactivo que pausa y ejecuta herramientas externas.

## Fase 4: Era Espectral y Ablations Unificados (V10 / V11 / serious_v1)
- [x] **Grid Search V10 Matrix-Free:** Demostración de escalado de rango Walsh $k$ (V10 dim512 k128 val_loss 3.9299).
- [x] **V11 Fourier-ALBERT:** Compartición de pesos de bloques virtuales (loss 4.1287 a 9.44M params).
- [x] **Run Serio A (`serious_v1`):** Ejecución de 2000 iters. Confirmada sensibilidad crítica al LR (1e-3 vs 1.5e-2 de V11).

## Fase 5: La Era Delta-Phase (V12 & Kernel C++/PyTorch en O(N)) — FRONTERA ACTUAL
Esta es la frontera activa del proyecto, transfiriendo la Regla Delta en Fase Compleja demostrada en `attention-neuron` (V298/V299).

### 🧪 Objetivos Técnicos V12
- [ ] **5.1. Kernel C++/PyTorch Vectorizado:**
  - Implementar prototipo de kernel fusionado en C++/PyTorch / TorchScript para el scan causal de la Regla Delta Matricial de Fase Compleja ($M_t = M_{t-1} + \frac{\beta}{d_k} e_t \otimes K_t$).
  - Benchmark de rendimiento en CPU (AVX-512) medido en **tokens/segundo en inferencia con memoria $O(1)$ constante**.
- [ ] **5.2. Arquitectura V12 (`model_spectral_v12_delta_phase.py`):**
  - Mezclador: `ShortCausalConv1D (k=4)` + `DeltaPhaseHolographicBlock` ($O(N)$).
  - Integración con embeddings factorizados y weight-tying de V11.
- [ ] **5.3. Pre-entrenamiento Serio V12:**
  - Ejecutar entrenamiento con sweep de LR (0.005 a 0.015) para romper la barrera de loss de V11 (4.12) en preentrenamiento de lenguaje.

---
*Roadmap actualizado por TinyThinker Architect. La memoria de fase en O(N) es el camino al Agente Soberano.*
