# 🗺️ Research Roadmap V10: La Frontera Matrix-Free

**Estado Actual:** Hemos logrado un modelo de lenguaje con asintótica paramétrica **$O(k^2)$**, normalización esférica (nGPT con *Spherical Loss*) y **Caché Espectral $O(1)$** (Fourier Hippocampus).

¿Hacia dónde va la investigación ahora? Estos son los 4 pilares fundamentales para el futuro del proyecto Tiny-Thinker:

---

## 1. Las "Chinchilla Scaling Laws" de la Inteligencia Espectral
Para convencer al mundo (y a nosotros mismos) de la robustez de esta arquitectura, necesitamos trazar su frontera matemática exacta.
- **Objetivo:** Definir cómo escala el Validation Loss frente a variaciones de $d$ (dimensión), $k$ (núcleo lógico) y $K_{mem}$ (frecuencias del hipocampo).
- **Hipótesis del Techo Lógico:** Sugerimos que la dimensión $d$ controla la "densidad de conocimientos" (vocabulario y hechos crudos), mientras que el ancho de banda $k$ controla el "IQ lógico" puro de la red. Una vez $k$ supera cierto umbral (ej. 128 o 256), subirlo más tiene rendimientos decrecientes.
- **Acción:** Realizar grid-searches metódicos en Colab GPU entrenando mini-modelos y graficar *Compute vs Loss*.

## 2. El Techo del Hipocampo (La Prueba del Millón de Tokens)
La memoria holográfica $O(1)$ funciona en contextos locales, pero debe someterse a la prueba de estrés industrial.
- **Objetivo:** Ejecutar la prueba *Needle In A Haystack* (La Aguja en el Pajar) a escalas masivas.
- **Procedimiento:** Alimentar a la red con el contenido equivalente a 10 libros técnicos por *streaming* (pasando bloques de tamaño $T_{chunk}$) y observar si, conservando únicamente las bajas frecuencias, el modelo es capaz de responder preguntas contextuales hechas en el primer libro.
- **Trascendencia:** Si la memoria semántica de las bajas frecuencias sobrevive a tal compresión, acabamos con la dictadura del "Attention Window" de OpenAI.

## 3. Fusión de Kernels en Triton / CUDA (SRAM Residency)
Si bien el $O(k^2)$ salva billones de parámetros, PyTorch genérico sigue teniendo que escribir tensores intermedios en la lenta VRAM (Global Memory) durante el forward pass.
- **Objetivo:** Alcanzar velocidades absurdas en inferencia (*throughput*).
- **Solución Hack:** Escribir un Kernel a medida en Triton que fusione las operaciones en cadena: cargar el tensor, aplicar la Fast Fourier Transform (FFT), la multiplicación por el `phase_gate` complejo, y el ensamblaje de la base de Walsh-Hadamard... todo dentro de la SRAM ultra-rápida de la GPU antes de devolver el resultado.
- **Impacto:** Podría viabilizar correr modelos LLaMA-Level nativos en un teléfono móvil con CPU convencional o NPU a miles de tokens por segundo.

## 4. Espectroscopía del Lenguaje (IA Interpretable al 100%)
La gran caja negra de la IA moderna son las matrices de pesos densos incomprensibles. Nuestra red, por el contrario, codifica todo en amplitudes y fases.
- **Objetivo:** Descifrar la estructura electromagnética/espectral del lenguaje humano.
- **Investigación:** Analizar qué "patrones de onda" (frecuencias) se activan en la capa `CausalComplexFFTMixer` cuando introducimos verbos vs. sustantivos. ¿Tienen los verbos de acción resonancias de alta frecuencia? ¿Son los conceptos abstractos perturbaciones de baja frecuencia?
- **Más Allá del NLP:** Dado que mapeamos los datos a física ondulatoria, esta arquitectura es la candidata ideal para modelar problemas **no textuales**:
  - Predicción del clima (fluctuaciones caóticas pero cíclicas).
  - Música generativa nativa (audio *end-to-end* sin tokenizadores).
  - Comportamiento bursátil o de fluidos.
