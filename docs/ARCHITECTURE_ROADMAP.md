# 🗺️ Architecture Roadmap & Session Synthesis

**Última Actualización:** 01 de Mayo de 2026
**Estado del Proyecto:** Transición exitosa a la "Era Espectral / Matrix-Free".

Este documento sirve como ancla de memoria y hoja de ruta para futuras sesiones de desarrollo en el proyecto `tiny-thinker`. Resume las arquitecturas punteras que hemos construido basándonos en los descubrimientos teóricos del repositorio `attention-neuron` y establece los próximos pasos accionables.

---

## 🧠 1. El Arsenal Arquitectónico (Lo que tenemos)

Hemos abandonado el paradigma clásico de los Transformers Densos ($O(N^2)$) para abrazar la compresión en el dominio de las frecuencias y la lógica de circuitos analógicos. Aquí está nuestro ecosistema actual:

### 1.1. `spectral_v4` (Matrix-Free Baseline)
*   **Concepto:** Reemplaza las inmensas matrices de pesos densos por proyecciones espectrales truncadas (DCT para Atención, Walsh-Hadamard para Feed-Forward).
*   **Logro:** Eliminó el *Memory Wall* de los parámetros. Demostró que un modelo de 18M (`dim=512`, `k=64/128`) sin *weight tying* puede arrasar a un modelo denso de 30M, alcanzando un `val_loss` de **3.33**. Es nuestro "Caballo de Batalla" estable.

### 1.2. `spectral_v5` (EXP-8: JPEG-LLM Cache)
*   **Concepto:** Compresión Temporal del Contexto Infinito. Aplica DCT a lo largo de la dimensión del tiempo (`seq_len`) del caché KV.
*   **Logro:** Permite descartar el 75% de la memoria del pasado (guardando solo la "onda semántica" de baja frecuencia). Hemos implementado **Noise-Injection** durante el entrenamiento para que la red sea robusta al leer sus propios recuerdos "borrosos".

### 1.3. `analog` (EXP-9: La Placa de Circuitos Evolutiva)
*   **Concepto:** Inspirado en los hallazgos V120 (Cosine Neurons). Destruye el Feed-Forward monolítico (ReLU/SiLU) dividiéndolo en 4 bancos paralelos:
    1.  **Lineal (SUM):** Semántica clásica.
    2.  **Multiplicativo (PROD):** Lógica AND estricta.
    3.  **Varianza (VAR):** Detección de anomalías.
    4.  **Periódico (SIN):** Resolución XOR / Alta frecuencia.
*   **Logro:** Récord absoluto del repositorio. Un modelo de apenas **15M de parámetros** destrozó la barrera del 3.0, logrando un `val_loss` de **2.99**. El bias inductivo matemático acelera el aprendizaje masivamente.

### 1.4. `coga_spectral` (Cognitive OS + Jerarquía V135)
*   **Concepto:** El "Sistema 2" de la IA. Implementa la jerarquía *Fast vs Slow Thinking*.
    *   **Cerebelo (Halt Head):** Decide si un token es fácil (lo escupe rápido) o difícil (entra en el bucle recurrente).
    *   **Bloque Core:** Inferencia multi-pasada con un **Scratchpad Mutable** (memoria RAM de trabajo) y un **Spectral MoE** (enrutamiento a especialistas de Walsh).
*   **Logro:** Con apenas **8.89M de parámetros**, logró un `val_loss` de **3.74**. Sacrifica la "reacción inmediata" en favor de la profundidad lógica y la deliberación.

### 1.5. `auto_architect` (Neurogénesis Residual V170)
*   **Concepto:** Neural Architecture Search (NAS) dinámico. La red empieza con 1 sola capa. Cuando se estanca, congela el pasado y "da a luz" a una nueva capa que solo aprende a corregir los errores (el residuo) de la anterior.
*   **Estado:** Programado y listo para su primer vuelo de prueba.

---

## ⚡ 2. El Optimizador SWO (`swo_optimizer/`)
Basado en los hitos V125 y V126 (Entropía Espectral Total).
*   **Qué hace:** La clase `SmoothAdam` comprime los estados históricos del gradiente (`m` y `v`) usando `adaptive_avg_pool2d` (reduciendo la resolución al 25% por eje).
*   **Impacto:** Reduce el consumo de RAM durante el entrenamiento en un **93.7%**, permitiendo entrenar modelos gigantescos en hardware de consumo. La "pérdida de resolución" actúa como un denoiser de gradientes muy potente.
*   **Estado:** Extraído a un repositorio *standalone* listo para publicar en GitHub con su propio `setup.py` y ejemplos.

---

## 🚀 3. Siguientes Pasos (The To-Do List)

Para la próxima vez que abramos el editor, aquí están los vectores de ataque más prometedores:

1.  **Evaluación de Inteligencia (IQ Test):**
    *   **Acción:** Ejecutar `scripts/eval.py` sobre los checkpoints de `analog_nano` y `coga_spectral_nano` usando el Golden Dataset.
    *   **Objetivo:** Cuantificar el *Tool-Calling Accuracy*. ¿Es el modelo Analógico más "listo" lógicamente que el V4 a pesar de ser más pequeño? ¿El Scratchpad de COGA le da ventaja en contextos difíciles?

2.  **Validación del JPEG-LLM (EXP-8):**
    *   **Acción:** Lanzar `train_spectral_jpeg_cache.yaml`.
    *   **Objetivo:** Comprobar si el *Noise-Injection* funciona y la red logra un *Loss* sub-4.0 mientras suprime el 75% del caché temporal.

3.  **El Vuelo de la Neurogénesis (Auto-Architect):**
    *   **Acción:** Modificar el bucle en `train.py` para que, si el *Loss* no mejora en 500 iteraciones, invoque `model.add_residual_layer()` automáticamente.
    *   **Objetivo:** Ver crecer a la red en vivo sin olvido catastrófico.

4.  **El Experimento "Mega-Layer" (Límites Físicos):**
    *   **Acción:** Crear `train_spectral_mega_64k.yaml` (dim=65536, k=128) y entrenarlo usando el flag `--optimizer swo`.
    *   **Objetivo:** Demostrar que podemos instanciar el espacio latente de un modelo frontera (GPT-3/Llama-3) en un ordenador portátil sin reventar la VRAM.

5.  **Futuro Lejano - Multimodalidad 3D DCT:**
    *   **Acción:** Extender las capas DCT para ingerir tensores 3D (parches de vídeo/imágenes) directamente en el dominio de las frecuencias, creando un "TinyThinker-Vision" sin usar pesadas redes convolucionales o ViTs densos.