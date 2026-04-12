# 📋 Plan de Mejora - TinyThinker

> Plan priorizado para evolucionar TinyThinker de un prototipo funcional a una base de código de investigación reproducible y escalable.

Última actualización: 8 de Abril de 2026 (Fase 5 Completada)

## ✅ Fase 4: Optimización y Re-nacimiento (Completado)
... (anteriormente completado) ...

## ✅ Fase 5: Escalamiento de Memoria (Completado)
Se ha superado la barrera de los 256 tokens para permitir flujos agénticos reales.

1.  **Contexto 1024:** Ampliación nativa de la ventana de inferencia.
2.  **Adaptación de RoPE:** Estabilización de embeddings posicionales en secuencias largas.
3.  **Chat Robusto:** Bucle de inferencia optimizado con KV-cache y manejo de inyección.

## 🚀 Fase 6: Capacidad y Coherencia (Inmediato)
Objetivo: Reducir las alucinaciones y mejorar la síntesis de información.

### P0 - Evaluación de Calidad
1.  **Pruebas de Estrés:** Identificar por qué el modelo de 12M pierde coherencia tras búsquedas largas.
2.  **Refinamiento de SFT:** ¿Necesitamos un dataset de fine-tuning más variado?

### P1 - Escalamiento de Parámetros (Escala B)
1.  **Entrenamiento 50M Params:**
    *   Usar `configs/train_scale_b.yaml`.
    *   Evaluar el impacto en la latencia de CPU.
    *   Comprobar si la mayor capacidad soluciona el "ruido" en las respuestas largas.

## 🚀 Fase 7: Arquitectura Avanzada y Cognitiva (COGA)
Objetivo: Implementar eficiencia arquitectónica para superar la limitación teórica de parámetros asumiendo el mismo coste computacional.

### ✅ P0 - Base Arquitectónica COGA (Completado)
1.  **Modularidad (MoE):** Creada arquitectura de "Expert Slots" (`model_moe.py`) para evitar olvido catastrófico.
2.  **Memoria de Trabajo:** Implementado RAM editable (`Scratchpad`) sin consumo de contexto (`model_coga.py`).
3.  **Sabiduría (MemoryBank):** Creada base de datos vectorial nativa para recuperación a largo plazo (`memory.py`).
4.  **Profundidad Adaptativa:** Implementado bucle Universal Transformer y estimador `Halt Head` para adaptar compute por token.
5.  **Autonomía:** Desarrollado script de ciclo nocturno para auto-finetuning basado en errores detectados en logs diarios.

### P1 - Validación Empírica COGA
1.  **Entrenamiento MoE Base:** Entrenar MoE y comparar con el baseline denso (actualmente en curso).
2.  **Validación de Componentes:** Probar empíricamente el Scratchpad, RAG Dinámico y Paro Adaptativo según el plan `docs/COGA_VALIDATION_PLAN.md`.
3.  **Currículum de Razonamiento:** Reanudar el entrenamiento de las fases lógicas (L2, L3) usando la nueva arquitectura COGA para enseñar al modelo a utilizar `<WRITE>`.

### P2 - Refactorización y Soporte
1.  **Soporte Universal:** Consolidar el entrenamiento (`train.py`) y chat (`chat.py`) para soportar dinámicamente cualquier arquitectura (dense, moe, coga). *(Completado)*

## 🧪 Notas de MLOps
- **Checkpoint Actual:** `ckpt_sft_latest.pt` (Basado en corpus 305M).
- **Log Estándar:** `[HH:MM:SS]` relativo al inicio del script.
- **Hardware Recomendado:** CPU Ryzen 7 8845HS (forzar `--device cpu`).

## 🛠 Backlog Secundario
- **DDP Support:** Preparar el código para entrenamiento distribuido (futuro).
- **Quantization:** Probar exportación a GGUF/INT8 para mayor velocidad.
- **Web UI:** Crear una interfaz sencilla para interactuar con el modelo agéntico.
