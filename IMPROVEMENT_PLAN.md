# 📋 Plan de Mejora - TinyThinker

> Plan priorizado para evolucionar TinyThinker de un prototipo funcional a una base de código de investigación reproducible y escalable.

Última actualización: 8 de Abril de 2026 (Fase 4 Completada)

## ✅ Fase 4: Optimización y Re-nacimiento (Completado)
Se ha reconstruido el corazón del modelo para solucionar problemas estructurales de base.

1.  **Corpus Masivo (305M Tokens):** Migración de cuentos infantiles a un mix con Wikipedia.
2.  **Tokenizador ByteLevel:** Solución definitiva al error de los espacios en blanco.
3.  **SFT con Enmascaramiento:** El modelo ya no repite el prompt, solo aprende a responder.
4.  **Hardware Ryzen (AVX-512):** Optimización 14x respecto a la iGPU para entrenamiento en CPU.
5.  **Metodología GEMINI.md:** Establecimiento de estándares de nombrado, backup y logging.

## 🚀 Fase 5: Escalamiento y Contexto (Inmediato)
Objetivo: Superar las limitaciones de memoria y capacidad identificadas en la Fase 4.

### P0 - Estabilidad de Inferencia
1.  **Ampliación de Contexto (1024/2048):**
    *   Actualizar `ModelArgs` y regenerar `freqs_cis` en `model.py`.
    *   Verificar estabilidad del RoPE en secuencias largas.
    *   Evitar el `AssertionError` al inyectar resultados de búsqueda extensos.

2.  **Refinamiento del Chat:**
    *   Implementar un sistema de "ventana deslizante" o truncado inteligente del historial.
    *   Mejorar la robustez del parser de resultados de búsqueda.

### P1 - Escalamiento de Parámetros (Escala B)
1.  **Entrenamiento 50M Params:**
    *   Usar `configs/train_scale_b.yaml`.
    *   Evaluar el impacto en la latencia de CPU (Ryzen 7 debería manejarlo bien).
    *   Comprobar si la mayor capacidad reduce las alucinaciones factuales.

## 🧪 Notas de MLOps
- **Checkpoint Actual:** `ckpt_sft_latest.pt` (Basado en corpus 305M).
- **Log Estándar:** `[HH:MM:SS]` relativo al inicio del script.
- **Hardware Recomendado:** CPU Ryzen 7 8845HS (forzar `--device cpu`).

## 🛠 Backlog Secundario
- **DDP Support:** Preparar el código para entrenamiento distribuido (futuro).
- **Quantization:** Probar exportación a GGUF/INT8 para mayor velocidad.
- **Web UI:** Crear una interfaz sencilla para interactuar con el modelo agéntico.
