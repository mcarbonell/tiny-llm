# 🧠 TinyThinker: Arquitectura Cognitiva para LLMs Diminutos

Este documento resume las ideas clave y la visión estratégica para el desarrollo de TinyThinker (~50M-100M parámetros), enfocándose en **razonamiento puro sobre memorización masiva**.

## 1. Filosofía Central: "Atleta Olímpico, no Culturista"
En lugar de forzar a un modelo pequeño a memorizar datos del mundo (Wikipedia, hechos históricos), la arquitectura se centra en:
*   **Capacidad Algorítmica:** Dominar la lógica de flujo de los lenguajes de programación.
*   **Externalización de Conocimiento:** Usar herramientas internas (Python, APIs, Vector DB) como memoria externa.
*   **Pensamiento Atómico:** Tratar las operaciones de razonamiento como unidades fundamentales.

## 2. El Tokenizador Lógico (Keywords como Átomos)
Para reducir el ruido ortográfico y maximizar la eficiencia en modelos de bajo parámetro:
*   **Palabras Reservadas:** Inyectar keywords de programación (`if`, `while`, `else`, `import`, `return`, `interface`, `async`, etc.) como tokens individuales e indivisibles.
*   **Tokenización Numérica:** Forzar que cada dígito (0-9) sea un token único. Esto permite al modelo realizar aritmética "columna por columna" en lugar de memorizar números grandes.
*   **Operadores de Pensamiento:** Tokens específicos para la lógica de razonamiento en etiquetas `<think>`: `ASSERT`, `BECAUSE`, `CONTRADICTION`, `IF_THEN`, `VERIFY`.

## 3. Modelo Controlador (Arquitectura Agente)
El LLM de 50M no es una enciclopedia, es el **Nervio Central** de un sistema mayor:
1.  **Analista de Intenciones:** Descompone la instrucción del usuario en pasos lógicos.
2.  **Orquestador de Herramientas:** Si el conocimiento no está en sus pesos, genera una llamada a una herramienta externa (ej: `<calc>2+2</calc>`).
3.  **Inline Tool Fusion:** Integración de herramientas durante el *forward pass* del modelo, permitiendo que la respuesta se nutra de datos externos sin detener la inferencia (estilo hipocampo artificial).

## 4. Estrategia de Entrenamiento: Trazas de Ejecución
No entrenar solo en "Pregunta -> Respuesta", sino en **"Estado A -> Pensamiento -> Acción -> Estado B"**:
*   **Chain of Code:** Entrenar al modelo para que genere pseudocódigo interno en sus trazas de pensamiento.
*   **Dataset Sintético de Lógica:** Inyectar ejemplos de razonamiento formal (silogismos, lógica booleana, diagramas de flujo) donde el éxito dependa de seguir reglas estrictas, no de recordar hechos.

## 5. Mantenimiento y Escalabilidad
*   **Expertos Vírgenes (MoE):** Dejar expertos vacíos en una arquitectura Mixture of Experts para futuros fine-tunes en dominios específicos, evitando el olvido catastrófico del núcleo lógico.
*   **Consolidación de Sueño (Offline):** Proceso de mejora donde el modelo analiza sus propios razonamientos pasados, un verificador los puntúa, y el modelo se auto-ajusta sobre sus mejores "trazas de pensamiento".

---
> [!NOTE]
> *Este documento es un borrador vivo basado en el brainstorming de Abril 2026. El objetivo final es un modelo que "piense" antes de hablar y sepa delegar lo que no conoce.*
