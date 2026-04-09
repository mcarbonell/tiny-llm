# 🏆 PROJECT STATUS: FASE 6 - ESCALADO A 50M (SCALE B)
**Fecha de actualización:** 8 de Abril de 2026
**Hito Principal:** Entrenamiento SFT del modelo de 12M verificado con éxito estructural. Inicio de pre-entrenamiento 50M.

## Estado de la Fase 5: Escalamiento de Memoria y SFT ✅ (Completada)
Se finalizó el *Supervised Fine-Tuning* (SFT) del modelo 12M utilizando el `dataset_golden_v1.json` (806 muestras) y ventana de 1024 tokens.
* **Resultado Analítico:** El modelo de 12M (Loss = 2.34) ha dominado el formato y comportamiento de agente (etiquetas `<THINK>`, `<TOOL_CALL>`). Las alucinaciones detectadas (p. ej. inventar "Lahorean Fejilin") certifican que el modelo es estructuralmente brillante pero cognitivamente pequeño.
* **Conclusión:** El software de inferencia adaptado funciona. El modelo está "aprendido" sintácticamente, pero los 12 Millones de parámetros limitan el entendimiento semántico.

## Fase Actual: Fase 6 - Capacidad y Coherencia 🚀 (En Progreso)
Objetivo: Entrenar el modelo Scale B (50M de parámetros). Al confirmar que la arquitectura y el pipeline de datos funcionan en la subescala, la capacidad de Scale B solucionará el déficit cognitivo y las alucinaciones.

### Trabajos Activos:
- Modificados `train.py` y `eval.py` para soportar carga dinámica de ficheros `.yaml` y contexto inteligente.
- Checkpoints de Escala A asilados en `/checkpoints/scale_a_12m/` para dejar espacio a la Escala B.
- Lanzando entrenamiento de 50 Millones de parámetros con aceleración DirectML (`configs/train_scale_b.yaml`).

## Métricas Actuales (Target Escala B)
- **Modelo:** ~50M Parámetros (`dim=512`, `n_layers=12`, `n_heads=8(GQA)`).
- **Contexto:** 1024 Tokens.
- **Corpus:** `train_combined.bin` (~300M tokens de TinyStories 1M + SimpleWiki).
