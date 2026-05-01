# Blueprint: JPEG-LLM Cache (Temporal DCT Compression)

## 1. El Cuello de Botella del Infinito
Para que un LLM tenga contexto "infinito" (pueda leer un libro entero de 1 millón de palabras y responder preguntas sobre él), el modelo debe almacenar en memoria RAM los vectores Key ($K$) y Value ($V$) de **cada uno de los tokens leídos**.
- En un Transformer clásico, el tamaño del Caché KV escala **linealmente** con la longitud de la secuencia: $O(L)$.
- Un libro de 1 millón de tokens en FP16 con un modelo medio exige cientos de Gigabytes de RAM unificada, bloqueando el avance del contexto infinito en hardware de consumo.

## 2. El Origen: "Text JPEG" (Findings V65)
En el repositorio `attention-neuron`, descubrimos que el lenguaje no es ruido sintáctico aislado, sino una "onda semántica" continua. Si aplicamos la **Transformada Discreta del Coseno (1D-DCT)** a una secuencia de embeddings de palabras a lo largo del **eje temporal**, y eliminamos las frecuencias altas (las de detalle), el modelo pierde la exactitud sintáctica (pierde artículos, adjetivos exactos) pero **retiene perfectamente la estructura lógica e intencional** del párrafo.

## 3. Arquitectura del Experimento 8 (JPEG-LLM Cache)

En `model_spectral_v5.py`, hemos implementado la compresión temporal directa en la capa `SpectralAttentionV5`. El algoritmo funciona así:

### Fase de Compresión (Almacenamiento)
Cuando el modelo termina de calcular $K$ y $V$ para una secuencia de tamaño $S$, en lugar de guardarla cruda:
1. Extrae las primeras $k$ filas (frecuencias bajas) de una base ortogonal DCT de tamaño $S \times S$, obteniendo $D_{k}$.
2. Transforma el tensor (pasando el eje del tiempo a la dimensión a multiplicar).
3. Obtiene el Caché Comprimido: $K_{freq} = K_{time} \cdot D_{k}^T$.
4. El tamaño del caché almacenado en RAM pasa de ser $S$ (ej. 256 tokens) a $k$ (ej. 64 coeficientes). Es una reducción plana y masiva del **75% del uso de memoria** instantáneamente.

### Fase de Descompresión (Recuperación)
Cuando llega un nuevo *query* (el usuario hace una pregunta sobre el texto pasado):
1. El modelo saca el $K_{freq}$ de 64 elementos de la RAM.
2. Invierte la transformada (multiplicando por la matriz DCT correspondiente) para "hinchar" la onda semántica de nuevo a sus 256 "tokens aproximados" originales ($K_{approx} = K_{freq} \cdot D_k$).
3. Añade el nuevo token a la cola.
4. Ejecuta un *Scaled Dot Product Attention* estándar, atendiendo sobre el recuerdo "difuso pero estructuralmente correcto" del documento.

## 4. Consecuencias del Experimento
Si el modelo `coga_spectral_v5` logra alcanzar pérdidas de validación similares a la versión sin compresión temporal, habremos demostrado que **es posible emular la retención semántica a largo plazo de un cerebro humano** (que recuerda conceptos e ideas, no palabras textuales), desbloqueando contextos de contexto virtualmente infinito limitados únicamente por las frecuencias matemáticas de compresión, no por el ancho de banda del hardware.
