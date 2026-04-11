# TinyThinker-COGA Implementation Roadmap

Este documento detalla el plan de acción para transformar el modelo **TinyThinker** (actualmente un modelo de lenguaje denso) en una implementación real de la arquitectura **COGA (Cognitive Operating System Architecture)**.

## Fase 1: Cimentación y Modularidad (MES - Expert Slots)
*Objetivo: Preparar al modelo para aprender sin olvidar.*

1.  **Refactorización MoE:** Migrar de una arquitectura densa a una arquitectura **MoE (Mixture of Experts)** ligera.
    *   Implementar la lógica de **Reserved Slots** (Expertos 13-16 bloqueados inicialmente).
    *   Configurar el `Router` para que ignore los slots reservados durante el pre-entrenamiento.
2.  **Protocolo de Inserción:** Crear scripts de entrenamiento que permitan congelar los "Expertos Base" y entrenar solo un "Expert Slot" con un dataset específico.
3.  **Validación:** Medir el "Olvido Catastrófico". El score en el dataset general debe mantenerse idéntico (+/- 0.1%) tras activar un nuevo experto especializado.

## Fase 2: El Motor de Deliberación (Scratchpad Mutable)
*Objetivo: Desacoplar el pensamiento de la emisión de tokens.*

1.  **Interfaz de Memoria de Trabajo:** Modificar la arquitectura para incluir un tensor externo (`scratchpad_tensor`) que persista durante los pasos de inferencia.
2.  **Tokens de Control (Primitivas CRUD):** Añadir al vocabulario tokens especiales: `<WRITE>`, `<EDIT>`, `<DELETE>`, `<READ>`.
3.  **Entrenamiento de "Pensamiento Invisible":**
    *   Generar datasets sintéticos donde la solución requiere pasos intermedios que se escriben en el scratchpad.
    *   Usar RL (Reinforcement Learning) para premiar al modelo cuando usa el scratchpad para corregir un error interno antes de emitir la respuesta.

## Fase 3: Persistencia y Sabiduría (Memorias Programables)
*Objetivo: Eliminar la amnesia entre sesiones.*

1.  **Integración de Vector DB:** Implementar una capa de recuperación (RAG dinámico) integrada en el ciclo de inferencia.
2.  **Lógica `remember()` / `recall()`:**
    *   Entrenar al modelo para identificar cuándo una información es "digna de ser recordada".
    *   Implementar un "Buffer de Memoria Episódica" que guarde las últimas interacciones y las resuma antes de moverlas a la memoria a largo plazo.
3.  **Contexto Aumentado:** Las memorias recuperadas se inyectan como embeddings adicionales en las capas intermedias.

## Fase 4: Profundidad Adaptativa (Recurrencia y Budget)
*Objetivo: Inteligencia variable según la dificultad del problema.*

1.  **Capa Recurrente (Universal Transformer):** Envolver el bloque central de capas en un bucle que permita reaplicar los mismos pesos N veces.
2.  **Mecanismo de Halting (Halt Head):** Añadir una "cabeza de salida" que prediga la probabilidad de parar el razonamiento.
3.  **Controlador de Recursos (Budget Estimator):** Un pequeño modelo que analice el input y asigne un `max_recurrence_steps`.

## Fase 5: Autonomía y Optimización (Heartbeat y Ciclo Nocturno)
*Objetivo: El agente "vivo" que se mejora a sí mismo.*

1.  **Sistema de Heartbeat:** Proceso en segundo plano que dispare inferencias proactivas cuando el usuario no está interactuando.
2.  **Pipeline de Sueño (Auto-Finetune):**
    *   Script que analiza los `logs/` del día para identificar fallos.
    *   Genera pares de entrenamiento SFT corregidos.
    *   Lanza un entrenamiento ligero en un **Expert Slot** durante la noche.
3.  **Consolidación:** Fusión de memorias redundantes y limpieza del banco de datos.

---
**Nota de Versiones:** Mantendremos versiones separadas del modelo (`model_dense.py`, `model_moe.py`, etc.) para garantizar la trazabilidad y comparabilidad de los experimentos.
