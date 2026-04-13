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
- **Sintético Lógica (10%) + Sintético Plan (5%):** Currículum de razonamiento niveles 0-4 y planificación.

### 3. Estado del Modelo (78M params)
- **Status:** Entrenando. Entrenamiento V1 del modelo 78M activo localmente (ver log más reciente en `logs/`).
- **Nuevo Objetivo:** Entrenar TinyThinker 78M desde cero usando el Dataset V1.
- **Scaling Law Warning:** Identificado que 108M tokens es insuficiente para 78M parámetros (Ratio 1.38:1). Este entrenamiento servirá como "Warmup" y validación de pipeline. El objetivo V2 será alcanzar 2B+ tokens.

### 4. Resultados del Sweep y Validación de Scaling Laws
Se han realizado barridos y entrenamientos completos de 5000 iteraciones para modelos más pequeños en la GPU L4 de Modal, demostrando empíricamente las *Scaling Laws*:
- **Nano (10M):** Perplejidad **244.95** (Ratio Tokens/Params = 10.3:1) - LR Óptimo: 1e-3, Warmup: 1000.
- **Micro (20M):** Perplejidad **158.70** (Ratio Tokens/Params = 5.4:1) - Gran salto de rendimiento.
- **Mini (30M):** Perplejidad **152.97** (Ratio Tokens/Params = 3.6:1) - Rendimiento marginal sobre el 20M debido al estrangulamiento de datos (data starvation).

Estas métricas validan la urgencia de expandir el dataset para la V2. El modelo de 78M (Ratio 1.38:1) se va a preentrenar fuertemente sub-alimentado de datos para validar la estabilidad arquitectónica final a gran escala.

## Próximos Pasos 🏁
1. **Lanzar Entrenamiento V1:** Validar que el modelo genera inglés coherente y respeta los tags `<think>`.
2. **Expandir Dataset a V2:** Aumentar el volumen de FineWeb-Edu y Cosmopedia para alcanzar el ratio de Chinchilla (1.5B tokens).
3. **Subir Context Window:** Aprovechar Flash Attention para pasar de 1024 a 2048 o 4096 tokens de contexto.

## Métricas finales de hoy:
- **Dataset V1 Tokens:** 107,919,573 tokens.
- **Vocabulario:** 16,384 (Versionado en `tokenizer_v1.json`).
- **Hardware Activo:** Radeon 780M (DirectML).
