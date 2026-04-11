# Plan de Validación Empírica - TinyThinker COGA

Dado el alto volumen de modificaciones estructurales, no podemos entrenar todo el modelo de golpe. Usaremos una estrategia de **"Ablación Constructiva"**: empezaremos con el modelo más simple y activaremos fases una a una, validando empíricamente cada mejora.

## Fase 1 Empírica: Validación del Router MoE (Phase 1)
**Objetivo:** Demostrar que el modelo puede retener conocimiento general mientras aprende una tarea especializada usando un *Expert Slot*.
1. **Pre-entrenamiento Base (Denso vs MoE):**
   - Entrenar `TinyThinker` (denso) durante `X` iteraciones en el dataset general. Guardar checkpoint.
   - Entrenar `TinyThinkerMoE` (con slots reservados bloqueados) durante las mismas `X` iteraciones en el mismo dataset.
   - *Criterio de éxito:* El *loss* de validación general del MoE debe ser igual o mejor que el denso.
2. **Prueba de "Olvido Catastrófico":**
   - Hacer un fine-tuning del modelo denso en un dataset muy especializado (ej. Lógica Matemática Pura). Evaluar su score en el dataset general (debería haber empeorado por el olvido).
   - Hacer el fine-tuning del `TinyThinkerMoE` desbloqueando el *Expert Slot 0* y re-entrenando solo el Router y ese Slot.
   - *Criterio de éxito:* El MoE domina la tarea matemática manteniendo su score exacto en el dataset general.

## Fase 2 Empírica: Insección del Scratchpad (Phase 2)
**Objetivo:** Enseñar al modelo a pensar en "voz baja" y usar sus tokens CRUD.
1. **Generación de Dataset Sintético (Trace-based):**
   - Usar un LLM frontera (ej. Claude o GPT-4) para generar un dataset de problemas complejos resueltos paso a paso, insertando manualmente etiquetas `<WRITE>...<END_WRITE>`.
2. **Supervised Fine-Tuning (SFT) de Primitivas:**
   - Entrenar el `TinyThinkerCOGA` (con MoE ya entrenado) usando este dataset sintético.
   - Modificar el bucle de pérdida (Loss) para que el modelo no sea penalizado por lo que escribe en el scratchpad, sino solo por la respuesta final.
   - *Criterio de éxito:* Durante la inferencia con `chat_coga.py`, el modelo invoca autónomamente `<WRITE>` antes de dar respuestas a preguntas complejas.

## Fase 3 Empírica: RAG Dinámico y Memoria (Phase 3)
**Objetivo:** Demostrar que la inyección de embeddings afecta positivamente la respuesta sin modificar los pesos.
1. **Test de Contexto Ciego:**
   - Preparar un set de preguntas sobre "hechos inventados" (ej. "La capital de Marte es Zorgon"). El modelo fallará.
2. **Inyección en Inferencia:**
   - Cargar el hecho en el `MemoryBank`.
   - Ejecutar la pregunta. El sistema RAG inyectará el embedding en el slot 0 del scratchpad.
   - *Criterio de éxito:* El modelo responde "Zorgon" sin haber sido re-entrenado, demostrando que su *Cross-Attention* lee correctamente el scratchpad.

## Fase 4 Empírica: Paro Adaptativo (Phase 4)
**Objetivo:** Validar que las tareas difíciles reciben más compute automático.
1. **Entrenamiento del Halt Head:**
   - Modificar el loop de entrenamiento. Para cada batch, forzamos al modelo a hacer 1, 2, 3 y 4 iteraciones del core. Medimos la pérdida (Loss) en cada iteración.
   - Entrenamos el `Halt Head` para predecir en qué iteración la mejora del Loss se estanca.
2. **Validación de Eficiencia:**
   - Pasar un dataset mixto (50% fácil, 50% difícil).
   - *Criterio de éxito:* El histograma de iteraciones del `core` debe mostrar un pico en 1 iteración para las tareas fáciles y un pico en 4 iteraciones para las difíciles.

---
**Nota Operativa:** Para ejecutar esto sin gastar cientos de horas de GPU, usaremos nuestro vocabulario reducido (16K) y entrenaremos en el dataset TinyStories o en la Wikipedia reducida del proyecto, limitando el pre-entrenamiento a unas pocas horas en la GPU local.