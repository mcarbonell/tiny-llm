# 📋 Plan de Mejora - TinyThinker

> **Contexto Evolutivo:** Este plan guía la transformación de las capacidades externas de **SOMA** y el rigor sistémico de **COGA** en un motor de inferencia espectral de última generación. El objetivo es un prototipo funcional, reproducible y escalable que demuestre inteligencia de frontera en hardware local.

Última actualización: 3 de Mayo de 2026 (Fase 7 Iniciada)

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
... (anteriormente completado) ...

### ✅ P1 - Validación y Estabilización Spectral V7 (Nuevo)
1.  **Factorización Exitosa:** Implementación de embeddings factorizados y cabezal vinculado para eliminar el "impuesto al vocabulario".
2.  **SMO Spectral-Aware:** Refactorización del optimizador para soportar núcleos DCT/Walsh nativos.
3.  **Estabilización de Init:** Identificación del umbral de inicialización (std=0.02) para modelos de alta dimensión (dim=1024).

### P2 - Arquitectura Avanzada: MoE Espectral
Objetivo: Combinar la eficiencia de los núcleos espectrales con la capacidad de los Expertos Modulares (MoE).
1.  **Spectral-MoE:** Implementar capas `WalshLinear` y `DCTLinear` dentro de bloques MoE con Gating Gumbel-Softmax.
2.  **Currículum de Razonamiento:** Reanudar el entrenamiento de las fases lógicas (L2, L3) usando la nueva arquitectura V7 para enseñar al modelo a utilizar `<WRITE>`.

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
