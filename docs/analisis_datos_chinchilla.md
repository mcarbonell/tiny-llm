# Análisis de Datos: Espectral vs Denso

### 1. El Equivalente en Parámetros Densos
La pregunta de Chinchilla para redes espectrales es clave.
- Tu modelo actual v8.6 `midi` tiene una dimensión $D=4096$ y 12 capas.
- **En un Transformer denso:** Una capa de atención y MLP de $D=4096$ tendría al menos $4 \times D^2 \approx 67$ millones de parámetros. Con 12 capas, sería un modelo de **~800 millones de parámetros**.
- **Tu modelo espectral:** Logra un flujo de información equivalente (gracias a la FWHT global) usando solo **17.9 millones**.

### 2. La Ley de Chinchilla para Espectrales
La regla de Chinchilla ($Tokens \approx 20 \times Parámetros$) se diseñó para redes densas porque cuenta los parámetros como un proxy del "ancho de banda cognitivo".
En tu caso, el ancho de banda no viene dado por tus 17.9M, sino por el "espacio de representación" de la FWHT (que equivale a los 800M de una red densa).
Por lo tanto:
- Chinchilla para 17.9M $\rightarrow$ ~360 Millones de tokens.
- **Chinchilla para el "Equivalente Denso" (800M)** $\rightarrow$ **~16 Billones (16B) de tokens**.

### 3. Tu Dataset Actual
He revisado `data/train_v2_32k.bin`:
- Pesa **193 MB**, lo que con `uint16` equivale a **~96 millones de tokens**.
- Estás entrenando una arquitectura con capacidad para 800M de parámetros usando un dataset que no llena ni el 1% de lo que Chinchilla recomendaría para ese tamaño.

**Conclusión:** Tu modelo está "hambriento" de datos. Está aprendiendo gramática básica pero no tiene suficientes textos para asimilar conocimiento general antes de memorizar y sobreajustarse al dataset (por eso el validation loss se estanca).

---

## 📚 Dónde conseguir Tokens de Calidad (HuggingFace)

Para dar el salto a un entrenamiento serio de la v8.6, necesitas un dataset de al menos **2 a 5 Billones (2B - 5B) de tokens** muy limpios.

### 1. FineWeb-Edu (El mejor para calidad)
HuggingFace liberó `HuggingFaceFW/fineweb-edu`. Es la versión purificada de la web, filtrada mediante modelos de IA para contener solo texto de nivel educativo (ciencia, historia, literatura).
- **Por qué es ideal:** Elimina el ruido de internet. Para tu modelo pequeño, el ruido es letal.

### 2. Cosmopedia / SmolTalk
`HuggingFaceTB/smoltalk` o Cosmopedia son datasets 100% sintéticos (generados por Mixtral/GPT-4).
- **Por qué es ideal:** Tienen una gramática perfecta y una lógica muy estructurada. Modelos como TinyLlama o Phi (modelos enanos que funcionan genial) se entrenan casi en exclusiva con datos sintéticos para evitar la basura de la web.

### 3. Cultura-X (Para español)
`uonlp/CulturaX` es el corpus de referencia multilenguaje. Puedes extraer la parte de español, que está muy bien limpiada (elimina deduplicaciones).

### 🛠️ Estrategia de Mejora de Dataset
En lugar de mezclar 10 archivos de texto como estás haciendo ahora en `build_master_corpus_v2.py` (que te da 400MB en total), el siguiente paso profesional sería:
1. Usar la librería `datasets` de HuggingFace.
2. Descargar 10GB de `FineWeb-Edu` (inglés) y 5GB de un buen corpus en español.
3. Mezclarlos y tokenizarlos directamente a `.bin` saltando el paso del archivo de texto intermedio.

¿Quieres que te prepare el script para descargar y tokenizar automáticamente un chunk masivo (ej. 1 Billón de tokens) de FineWeb-Edu para tu próxima iteración?