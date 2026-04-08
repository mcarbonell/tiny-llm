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

## 🧪 Notas de MLOps
- **Checkpoint Actual:** `ckpt_sft_latest.pt` (Basado en corpus 305M).
- **Log Estándar:** `[HH:MM:SS]` relativo al inicio del script.
- **Hardware Recomendado:** CPU Ryzen 7 8845HS (forzar `--device cpu`).

## 🛠 Backlog Secundario
- **DDP Support:** Preparar el código para entrenamiento distribuido (futuro).
- **Quantization:** Probar exportación a GGUF/INT8 para mayor velocidad.
- **Web UI:** Crear una interfaz sencilla para interactuar con el modelo agéntico.
