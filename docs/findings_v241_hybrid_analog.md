# Hallazgos V241: Sinergia Híbrida (Adam + DGE)

## Objetivo
Optimizar la arquitectura híbrida para resolver el **Cumulative Modulo Challenge**. El objetivo era superar el estancamiento del 33% mediante una mejor coordinación entre el optimizador analítico (Adam) y el libre de gradientes (DGE).

## Resultados (POC V241)

| Métrica | Valor |
| :--- | :--- |
| **Precisión Final (Logic)** | **83.38%** |
| **Loss Final** | 0.74 |
| **Parámetros DGE** | 64 |
| **Mejora sobre azar** | **5.8x** |
| **Tiempo de Entrenamiento** | 13 min (CPU) |

## Avances Técnicos Clave

### 1. El Puente STE (Straight-Through Estimator)
La innovación principal fue el uso de un **STE Multivariable**. 
- Antes, Adam no podía entrenar las proyecciones lineales que alimentaban al módulo porque el gradiente era cero. 
- Al implementar el STE, Adam "ve" a través del módulo y puede alinear los embeddings para que el Banco Simbólico reciba los valores correctos.

### 2. Normalización de Bancos (Scale Balancing)
Se añadió una `LayerNorm` a la salida del Banco Simbólico. Esto evitó que los valores crudos del módulo (0 a 7) desestabilizaran la red frente a los otros bancos (Linear/Sin) que operan en rangos menores.

### 3. Reducción del Espacio de Búsqueda DGE
Al delegar la proyección a Adam, DGE solo tuvo que optimizar **64 parámetros** de ruteo lógico. Esto permitió subir el Learning Rate de DGE a **0.2** y el Delta a **0.5**, logrando que el modelo "saltara" entre las discontinuidades del módulo con total fluidez.

## Conclusión
Este experimento demuestra que la **Diferenciabilidad Mixta** es el futuro de la eficiencia en TinyThinker. No necesitamos que DGE lo haga todo; necesitamos que Adam sea los "ojos" (percepción) y DGE sea el "cerebro" (lógica). 

### Próximos Pasos
- Integrar este `AnalogFeedForwardHybrid` en el modelo **Spectral V8.3**.
- Usar DGE para el **ruteo de expertos (MoE)** en lugar de Softmax, permitiendo una sparsity real y dura sin pérdida de gradiente.
