# 🏆 PROJECT STATUS: CURRÍCULUM DE RAZONAMIENTO Y PLANIFICACIÓN
**Fecha de actualización:** 12 de Abril de 2026
**Hitos Principales:**
- ✅ **Lógica (L0-L4) Completada:** Generadas y dedupas 3,528 muestras únicas.
- ✅ **Planificación (COGA) Completada:** Generadas y dedupas 163 muestras multidominio.
- ✅ **Tooling de Datos:** Implementado deduplicador semántico (`MiniLM-L6-v2`) verificado.

## Estado de la Fase 6: Escalamiento a 50M ⏳ (En Pausa)
El modelo de 50M (Scale B) está en espera. Se ha decidido priorizar el "Child Logic Tune" (Fase 2) usando el nuevo dataset sintético antes de reanudar el pre-entrenamiento a gran escala.

## Fase Actual: Fase 2 - Fine-Tuning de Lógica y Agente 🚀 (Listo para Entrenamiento)
Objetivo: Entrenar al TinyThinker en razonamiento estructurado paso a paso y planificación de tareas mediante el nuevo currículum generado.

### Hitos Conseguidos hoy:
- **Expansión Temática:** Incrementada la variedad de temas en `generate_planning_samples.py` a ~50 por dominio.
- **Deduplicación Semántica:** Integrado `scripts/deduplicate_dataset.py` para asegurar la diversidad del dataset (umbral sim=0.92).
- **Planificación Multi-dominio:** Generadas muestras de alta calidad en *Agentic*, *Technical*, *Creative* y *Household*.

### Próximos Pasos (Pendiente de GPU):
1. **Consolidar Datasets:** Unir Lógica + Planificación en un único archivo de entrenamiento.
2. **Phase 2 SFT:** Lanzar el entrenamiento "Child Logic Tune" (Phase 2) con el nuevo corpus.
3. **Validación COGA:** Probar las primitivas de memoria y scratchpad en inferencia real.

## Métricas finales de hoy:
- **Logic Curriculum:** 3,528 muestras únicas.
- **Planning Batch:** 163 muestras únicas (de 200 originales).
- **Modelo Generador:** `google/gemma-4-31b-it:free` (Excelentes resultados en formato y lógica).
