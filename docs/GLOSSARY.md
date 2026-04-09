# 📖 Glosario Progresivo (TinyThinker / TinyLLM)

Este documento es una guía de referencia estructurada por niveles de dificultad. Sirve como una introducción paulatina a los conceptos de Inteligencia Artificial que hacen funcionar a TinyThinker.

## 🟢 Nivel Básico: Los Fundamentos

*   **LLM (Large Language Model / Modelo de Lenguaje Grande):** Un sistema de IA diseñado para comprender y generar lenguaje natural a gran escala. TinyThinker es un LLM en miniatura.
*   **Prompt y System Prompt (Prompt de Sistema):** El *Prompt* es la instrucción o pregunta del usuario. El *System Prompt* son directrices ocultas proporcionadas al modelo en el *backend* antes de interactuar (ej. "Eres un asistente de programación educado, responde en español").
*   **Token:** La unidad mínima de procesamiento para la IA. No equivale siempre a una palabra; frecuentemente son sílabas o fragmentos (ej. "Glos" y "ario").
*   **Tokenizador (Tokenizer):** El intérprete encargado de traducir nuestra cadena de texto en listas de identificadores numéricos (tokens) antes de enviarlo al modelo, y de des-tokenizar la respuesta de vuelta a texto legible.
*   **Ventana de Contexto (Context Window):** El límite de "memoria a corto plazo" del modelo. Si la ventana es de 2048 tokens, el modelo descartará internamente las instrucciones que queden más de 2048 fragmentos atrás en la conversación actual.
*   **Dataset (Conjunto de Datos):** La materia prima del entrenamiento. Los datasets de *Pre-entrenamiento* suelen ser enormes corpus de texto no estructurado (para enseñar gramática y cultura general). Los datasets de *Fine-tuning* son altamente depurados y se basan en formato "instrucción-respuesta" o casos de uso puros.

## 🟡 Nivel Medio: Arquitectura Core y Entrenamiento

*   **Transformer:** La arquitectura de red neuronal detrás de casi todos los LLM modernos. Destaca por procesar todo el bloque de texto aportando "rutas cortas" para no tener que leer la secuencia estrictamente palabra por palabra, como los modelos arcaicos.
*   **Capas (Layers / Transformer Blocks):** Bloques de procesamiento ensartados uno tras otro en el Transformer. En la capa baja se capturan similitudes básicas (gramática), mientras que capas más profundas procesan conceptos abstractos y contexto.
*   **Embedding (Incrustación):** La primera capa del Transformer. Convierte el número solitario de un token en un vector geométrico de cientos de dimensiones, posicionando conceptos parecidos cerca en un hiperespacio (las matemáticas entienden que "Perro" y "Gato" están próximos, "Avión" está lejos).
*   **Parámetros:** Los "pesos" matemáticos de la arquitectura que actúan como diales. Empiezan aleatorios, y se van ajustando lentamente durante el entrenamiento hasta codificar con éxito la lógica y el lenguaje.
*   **Entrenamiento (Época, Paso, Pérdida):** 
    *   **Paso (Step):** La acción de analizar un pequeño lote de texto (batch) y actualizar ligeramente los parámetros.
    *   **Época (Epoch):** Hacer tantos "steps" como sea necesario para haber leído el 100% de los archivos del dataset una vez.
    *   **Pérdida (Loss):** Valor que mide el nivel de equivocación prediciendo una palabra de ejemplo. Nuestro `GEMINI.md` pide medir esto con al menos cuatro decimales. A menor Loss, mayor precisión.

## 🔴 Nivel Avanzado: Optimizaciones y Técnicas Modernas

*   **Atención Multi-Cabezal (Multi-Head Attention):** El núcleo de la inteligencia del Transformer. Evalúa matemáticamente cuánta "importancia" prestarle a cada palabra restante de la oración. Al ser *Multi-Cabezal*, el modelo analiza diferentes perspectivas a la vez (una cabezal mira el género pronominal, otro cabezal mira el sujeto de la acción).
*   **Conexiones Residuales (Residual Connections):** Rutas que puentean y esquivan procesamiento pesado añadiendo matemáticamente la información de entrada original directo a la salida. Permite que no se destruya o se diluya la señal de un texto tras atravesar docenas de *Capas*.
*   **KV-Cache (Key-Value Cache):** Optimizacion que cambia las normas de juego durante la generación de texto (inferencia). Evita que en cada nueva palabra deducida el LLM tenga que recalcular todo el texto previo a fuerza bruta, guardando las salidas matriciales y limitándose a calcular el último token aportado.
*   **Checkpoint:** Instantánea persistida en disco (archivo `.pt` o `.safetensors`) del valioso estado de los Parámetros del modelo. Fundamental para reiniciar entrenamientos tras errores o probar evaluaciones a medias.
*   **LoRA (Low-Rank Adaptation):** Una técnica maestra de *Fine-Tuning*. En lugar de actualizar y entrenar todo el tamaño real de los parámetros del modelo (lo cual quema muchísima RAM / VRAM), **LoRA** inserta "parches" temporales y pequeños que son los únicos que se entrenan, abaratando sustancialmente el coste y sin perder casi precisión final.
*   **Tool Calling (Llamada a Herramientas, Funciones o AI Agent):** Técnica donde se "afina" el LLM para que en base a una petición devuelva texto en formatos máquina como JSON. Ese texto es interceptado por el código para pedir a software real un cálculo (Consultar el clima, evaluar una API, usar una calculadora) y el humano siente que el modelo está usando en vivo el software.
