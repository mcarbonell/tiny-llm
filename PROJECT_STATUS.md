# 🏆 PROJECT STATUS: REINICIO ESTRATÉGICO V1 - HIGH DENSITY
**Fecha de actualización:** 12 de Abril de 2026

## 🚀 Hito Principal: Pipeline V1 Completado
Se ha realizado un reinicio estratégico para solucionar problemas de desincronización entre el tokenizador y el modelo, y para modernizar el núcleo del entrenamiento.

### 1. Ingeniería de Software & Optimización
- ✅ **Flash Attention:** Integrado nativamente en todas las arquitecturas (`dense`, `moe`, `coga`). Permite escalar la ventana de contexto reduciendo drásticamente el uso de VRAM ($O(n)$).
- ✅ **Trazabilidad de Experimentos:** El script `train.py` ahora registra obligatoriamente en los logs el fichero del modelo, el tokenizador versionado y el dataset utilizado.
- ✅ **Versionado de Tokenizadores:** Implementada política de no-sobrescritura. Creado `tokenizer_v1.json` como primer estándar sólido.

### 2. Pipeline de Datos V1 (108M Tokens)
Se ha abandonado el uso de Wikipedia cruda a favor de un corpus de "Alta Densidad":
- **FineWeb-Edu (40%):** Filtrado por LLMs para máximo valor educativo.
- **Cosmopedia v2 (30%):** Conocimiento sintético estructurado perfecto.
- **TinyStories v2 (15%):** Narrativa simple con tags `<|endoftext|>` limpios.
- **Sintético Lógica (15%):** Currículum de razonamiento niveles 0-4.

### 3. Estado del Modelo (78M params)
- **Status:** El modelo anterior fue descartado por desincronización de vocabulario.
- **Nuevo Objetivo:** Entrenar TinyThinker 78M desde cero usando el Dataset V1.
- **Scaling Law Warning:** Identificado que 108M tokens es insuficiente para 78M parámetros (Ratio 1.38:1). Este entrenamiento servirá como "Warmup" y validación de pipeline. El objetivo V2 será alcanzar 2B+ tokens.

## Próximos Pasos 🏁
1. **Lanzar Entrenamiento V1:** Validar que el modelo genera inglés coherente y respeta los tags `<think>`.
2. **Expandir Dataset a V2:** Aumentar el volumen de FineWeb-Edu y Cosmopedia para alcanzar el ratio de Chinchilla (1.5B tokens).
3. **Subir Context Window:** Aprovechar Flash Attention para pasar de 1024 a 2048 o 4096 tokens de contexto.

## Métricas finales de hoy:
- **Dataset V1 Tokens:** 107,919,573 tokens.
- **Vocabulario:** 16,384 (Versionado en `tokenizer_v1.json`).
- **Hardware Activo:** Radeon 780M (DirectML).
