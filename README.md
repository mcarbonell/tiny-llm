# 🧠 TinyThinker (LLM Lógico Minimalista)

![Status](https://img.shields.io/badge/Status-Fase_1_Entrenando-blue?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Params](https://img.shields.io/badge/Params-12.46M_(pretraining)-lightgrey?style=for-the-badge)

Un experimento educativo para construir, entrenar y evaluar un Modelo de Lenguaje Pequeño/Grande experimental completamente desde cero usando PyTorch. El modelo de **pre-entrenamiento actual tiene 12.46M parámetros** (dim=256, 6 capas), escala pensada para validar el pipeline en CPU local. El objetivo final es escalar a **100M–300M parámetros** en entorno cloud.

El modelo se basa en la filosofía de **"Cerebro Pequeño, Lógica Fuerte"**, donde el razonamiento (*Chain of Thought*) prevalece sobre el almacenamiento masivo de datos factuales, apoyándose en la externalización de conocimientos usando la herramienta autónoma de búsqueda **(Tool-calling)**.

## ✨ Características Técnicas (Modernizadas)
En lugar de basarnos en el viejo paper de 2017, la arquitectura del Transformer se basa en los avances punta actuales:
- **RoPE (Rotary Position Embeddings):** Mejor extrapolación contextual.
- **SwiGLU:** Activaciones neuronales más densas en las capas ocultas (estilo LLaMa).
- **RMSNorm:** Mayor control, estabilidad temporal y rendimiento puro.
- **Grouped-Query Attention (GQA):** Reducción drástica del caché de Key/Value, ideal para correr las inferencias rápidamente en CPU locales.
- **Acumulación de Gradientes & Precisión Mixta:** Maximización de hardware modesto a través de tensores *BFloat16*.

## 🚀 Fases del Proyecto
1. **Fase 1 - Adquisición del Lenguaje:** Pre-entrenamiento inicial con datasets limpios focalizados (como `TinyStories`) para que la red adquiera gramática impecable sin intoxicarse.
2. **Fase 2 - Razonamiento (CoT):** Ajuste fino (*CPT*) en subconjuntos racionales para enseñar la planificación estructural mediante tags `<THINK>`.
3. **Fase 3 - Búsqueda de Herramientas:** Condicionamiento y loop interactivo a través de sintaxis estricta, detectando tags `<TOOL_CALL>` para conectar con APIs (ej. Wikipedia / Buscadores).

## 🛠️ Estructura del repositorio
- `/model`: Código de la arquitectura neuronal pura (`model.py`) y tokenizadores base.
- `/scripts`: Flujos de entrenamiento pesados, utilidades de procesamiento de bits y descargas.
- `/data`: Submódulo para almacenamiento binario transitorio **(Ignorado en GIT)**.
- `/tests`: Unit-tests de integridad matricial de PyTorch.
- `/docs`: Documentos de diseño subyacentes e ideas futuras.
- `/logs`: Histórico y analíticas de la evolución del hiperparámetro Loss durante los entrenamientos.

## ⚡ Quick Start
```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Descargar TinyStories y entrenar tokenizador BPE
python scripts/download_and_tokenize.py

# 3. Tokenizar y guardar el dataset en binario
python scripts/prepare_data.py

# 4. Arrancar el pre-entrenamiento
python scripts/train.py

# 5. Fine-tuning con LoRA
python scripts/finetune.py --lora_r 8 --lora_alpha 16 --data_file data/tool_dataset_real.json

# 6. (Opcional) Inferencia interactiva con tool-calling
python scripts/chat.py

# 7. Evaluar el modelo (perplexity y accuracy en tool-calling)
python scripts/eval.py --checkpoint checkpoints/ckpt_finetuned.pt
```

## 📌 Nota sobre dataset
El finetuning por defecto usa `data/tool_dataset_real.json`, pero puedes cambiar el archivo con `--data_file`.

---
*Construido artesanalmente desde cero.*
