# Blueprint: Proyecto "TinyThinker" (LLM Lógico Minimalista)

## 1. Filosofía y Objetivos del Proyecto
El objetivo de este proyecto **no** es crear un producto comercial ni utilizar un modelo pre-entrenado (como Llama o Mistral) para hacer fine-tuning. 
El objetivo es puramente **educativo y experimental**: construir, entrenar y evaluar un Modelo de Lenguaje Grande (LLM) completamente desde cero usando PyTorch.

**Características clave del modelo (TinyThinker):**
*   **Cerebro Pequeño, Lógica Fuerte:** Tendrá entre 100M y 300M de parámetros.
*   **Cero Memorización:** No queremos que memorice capitales, historia o datos factuales.
*   **Uso de Herramientas (Tool-Use):** Debe externalizar el conocimiento. Si no sabe algo, debe emitir tokens especiales para invocar una búsqueda en internet y leer el resultado antes de responder.
*   **Idioma:** Inglés (para aprovechar la abundancia de datasets de razonamiento de alta calidad).

## 2. Pila Tecnológica y Hardware
*   **Framework:** PyTorch (nativo).
*   **Entorno Local (Desarrollo y Testing):** AMD Ryzen 7 8845HS, 64GB RAM DDR5, GPU Integrada Radeon 780M. (Se usarán batches muy pequeños o CPU para validar que el código compila y el *loss* desciende).
*   **Entorno Cloud (Entrenamiento Pesado):** GPU alquilada (ej. RunPod/Vast.ai con 1x RTX 4090 o A6000) usando scripts generados aquí.

## 3. Arquitectura del Modelo (Modernizada)
No usaremos el Transformer clásico de 2017 (Vaswani et al.). Implementaremos desde cero técnicas de 2023/2024:
1.  **RoPE (Rotary Position Embeddings):** En lugar de embeddings posicionales absolutos.
2.  **SwiGLU:** Función de activación en la capa FeedForward (estilo Llama 3).
3.  **Grouped-Query Attention (GQA):** Para reducir el coste de memoria en la inferencia.
4.  **RMSNorm:** En lugar de LayerNorm estándar para mayor estabilidad y velocidad.
5.  **Tokenizador:** Entrenaremos un tokenizador BPE (Byte-Pair Encoding) desde cero usando un vocabulario pequeño (ej. 32,000 tokens) o usaremos `tiktoken`.

## 4. Pipeline de Datos (El secreto del éxito)
El modelo se entrenará en tres fases secuenciales:

*   **Fase 1: Adquisición del Lenguaje (Pre-training)**
    *   *Objetivo:* Aprender gramática, sintaxis y a hablar inglés fluido.
    *   *Dataset:* `roneneldan/TinyStories` (historias generadas con vocabulario de un niño de 4 años) + un subconjunto ínfimo y muy filtrado de `HuggingFaceFW/fineweb-edu`.
*   **Fase 2: Razonamiento (Chain of Thought)**
    *   *Objetivo:* Enseñar a pensar paso a paso antes de escupir una respuesta.
    *   *Dataset:* Subconjuntos filtrados de `Open-Orca/OpenOrca` o `gsm8k` (matemáticas básicas).
*   **Fase 3: Uso de Herramientas (Tool-Calling / Function Calling)**
    *   *Objetivo:* Enseñar la sintaxis `<TOOL_CALL>`.
    *   *Dataset:* **Generación Sintética**. El agente deberá escribir un script que use una API barata (ej. OpenAI gpt-4o-mini o Anthropic Haiku) para generar ~20,000 ejemplos con el siguiente formato estricto:
        `User: Who is the current president of France?`
        `Assistant: <THINK> I don't store factual data. I need to search the web. </THINK> <TOOL_CALL> search("current president of France") </TOOL_CALL> <TOOL_RESULT> Emmanuel Macron </TOOL_RESULT> Based on my search, the current president of France is Emmanuel Macron.`

## 5. Instrucciones para el Agente Autónomo (AI IDE)
Si eres un agente de IA leyendo este documento para inicializar el proyecto, sigue este plan de acción paso a paso. **No saltes al paso 2 sin terminar y probar el paso 1.**

*   **Paso 1: Setup del Entorno y Tokenizador**
    *   Crea la estructura de carpetas (`/model`, `/data`, `/scripts`, `/tests`).
    *   Escribe el script para descargar TinyStories.
    *   Entrena un tokenizador BPE personalizado sobre una muestra de TinyStories y guárdalo.
*   **Paso 2: Arquitectura (model.py)**
    *   Implementa el modelo Transformer con RoPE, RMSNorm, SwiGLU y GQA en PyTorch.
    *   Escribe tests unitarios (`pytest`) para asegurar que las dimensiones de los tensores coinciden en el forward pass.
*   **Paso 3: Bucle de Entrenamiento (train.py)**
    *   Escribe el loop de entrenamiento. Debe soportar Gradient Accumulation, Mixed Precision (bf16/fp16) y guardar checkpoints (`.pt`).
    *   Haz una prueba de entrenamiento local (100 iteraciones) para verificar que el loss disminuye.
*   **Paso 4: Generación Sintética de Datos (tools_dataset.py)**
    *   Escribe el script para generar el dataset de la Fase 3 usando llamadas asíncronas a una API (deja los placeholders de las API keys listos para el usuario).
*   **Paso 5: Inferencia y Bucle de Herramientas (chat.py)**
    *   Escribe un script de chat iterativo. Si el modelo genera `<TOOL_CALL>`, el script de Python debe detener la generación, extraer la query, hacer una petición real a Wikipedia/DuckDuckGo, empaquetar el resultado en `<TOOL_RESULT>` y reanudar la generación del modelo.

---
*Fin del Blueprint. Agente, por favor, confirma la lectura de este documento y sugiere el primer comando de terminal a ejecutar para comenzar con el Paso 1.*