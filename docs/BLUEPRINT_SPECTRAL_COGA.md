# Blueprint: Spectral COGA (Cognitive Operating System Architecture + Spectral Synthesis)

## 1. El Paradigma de la Computación Asimétrica
Los Transformers densos clásicos padecen de dos ineficiencias fundamentales:
1. **Homogeneidad de Inferencia (Feed-Forward rígdio):** Gastan la misma energía y parámetros para resolver "2+2" que para planificar un viaje espacial. Tienen exactamente $L$ capas de profundidad computacional, independientemente de la entropía de la tarea.
2. **Memorización vs. Razonamiento (El Memory Wall):** Sus pesos (e.g. 10B a 100B parámetros) son densos, lo que requiere inmensos anchos de banda de memoria RAM simplemente para intentar representar la lógica y la semántica, inflando los costes a niveles insostenibles.

**Spectral COGA** fusiona dos arquitecturas rompedoras para solucionar ambos problemas:
- Del proyecto `tiny-thinker` hereda **COGA**: Una arquitectura de sistema operativo con *Scratchpad Mutable* (working memory) e *Inferencia Recurrente* de múltiples pasadas.
- Del proyecto `attention-neuron` hereda la **Síntesis Espectral Matrix-Free**: La compresión de matrices masivas (Q,K,V,O y FFNs) en coeficientes frecuenciales (DCT y Walsh-Hadamard).

## 2. La Anatomía del Agente

La arquitectura `TinyThinkerCogaSpectral` divide el flujo cognitivo en tres fases, gobernadas por un "Cerebelo" central de Inferencia Dinámica:

### 2.1 Fase I: Bloque Pre (Parsing Sensorial)
- **Función:** Lectura del token de entrada y construcción de la semántica superficial.
- **Componentes:** Transformers espectrales ultraligeros. 
- **Atención:** `DCTLinear` (Compresión semántica suave mediante ondas coseno).
- **Lógica:** `WalshLinear` (Cortes binarios lógicos mediante ondas cuadradas).
- *No tiene acceso al Scratchpad.* Su coste computacional es mínimo.

### 2.2 Fase II: El Cerebelo Espectral (Early-Exit Gate)
Basado en los hallazgos de la neurona de atención (Experimento V89), antes de invocar la costosa "Reflexión", el modelo consulta una capa lineal simple (el *Halt Head*).
- Si la **Entropía Predictiva** de la salida pre-procesada es altísima (confianza nula), el Cerebelo activa el bucle recurrente del Bloque Core al 100%.
- Si la entropía es nula (el modelo está completamente seguro de la siguiente palabra trivial), la probabilidad de *Halt* se acerca a $1.0$, y el modelo **se salta la mayoría (o todas) las iteraciones recurrentes** ahorrando inmensas cantidades de FLOPs.

### 2.3 Fase III: Bloque Core (Razonamiento Profundo)
- **Función:** Pensamiento extendido (Chain-of-Thought endógeno) independiente de la emisión de tokens.
- **Recurrencia Adaptativa:** En lugar de apilar 100 capas físicas, apilamos 4 capas lógicas y el modelo "itera" sobre ellas hasta $N$ veces (según lo determine el Cerebelo).
- **El Scratchpad Mutable:** Una sección externa de memoria ($N$ slots vectoriales) a la que el modelo atiende (`SpectralCrossAttention`). En cada ciclo de recurrencia, puede reescribir hipótesis sin polucionar su "memoria a corto plazo" (el contexto).
- **Spectral MoE:** El Feed-Forward es un Mixture-of-Experts donde cada experto es un combinador Walsh-Hadamard. El enrutamiento es semántico, pero la computación interna de cada experto carece de multiplicaciones complejas, pudiendo escalar a decenas de expertos sin sobrecarga de VRAM.

### 2.4 Fase IV: Bloque Post (Refinamiento y Salida)
- Homologa la representación iterada final, estabiliza la distribución y la proyecta al vocabulario real mediante un cabezal de salida estándar separado de los embeddings de entrada (sin *Weight Tying* en modelos pequeños, para optimizar val_loss).

## 3. Impacto Teórico (Turing-Completo Edge AI)

Al combinar recurrencia (un bucle condicional `while`), memoria de trabajo mutable (Scratchpad `read/write`) e hiper-compresión matemática (DCT $O(N \log N)$), Spectral COGA no es un mero "predictor de secuencias". Es un **computador neural cuasi Turing-Completo** capaz de funcionar íntegramente en la CPU de un ordenador portátil estándar, revolucionando el Edge AI y la robótica autónoma.
