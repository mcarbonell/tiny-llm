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
2. **Fase 2 - Razonamiento (TinyLogic):** Entrenamiento progresivo mediante un currículum de 7 niveles (habilidades cognitivas crecientes) usando trazas de pensamiento `<think>`.
3. **Fase 3 - Agente Autónomo:** Integración de tool-calling para conectar con APIs externas.

## 🛠️ Estructura del repositorio
- `/model`: Código de la arquitectura neuronal pura (`model.py`) y tokenizadores base.
- `/scripts`: Flujos de entrenamiento pesados, utilidades de procesamiento de bits y descargas.
- `/data`: Submódulo para almacenamiento binario transitorio **(Ignorado en GIT)**.
- `/tests`: Unit-tests de integridad matricial de PyTorch.
- `/docs`: Documentos de diseño subyacentes e ideas futuras.
- `/logs`: Histórico de métricas y analíticas.

## 🎓 TinyLogic & Planning Curriculum
Hemos implementado un sistema de aprendizaje graduado inspirado en el desarrollo humano, consolidando un corpus sintético de alta diversidad:

- **Logic Pipeline (3,528 muestras):**
    - **L0-L1 (Foundation):** Categorización y lógica concreta simple.
    - **L2-L3 (Structured):** Lógica transitiva, secuencias y deducción multi-paso.
    - **L4 (Advanced):** Abstracción, razonamiento probabilístico y meta-lógica.
- **Planning Domain (163 muestras):**
    - **Agentic (COGA):** Razonamiento sobre primitivas de memoria, scratchpad y herramientas.
    - **Technical/Creative/Household:** Descomposición de tareas complejas en planes de ejecución lógicos.

**Deduplicación:** Todo el dataset ha sido procesado mediante `scripts/deduplicate_dataset.py` usando embeddings semánticos (`all-MiniLM-L6-v2`) para garantizar que la red no aprenda de redundancias sintéticas.

## ⚡ Quick Start
```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Construir tokenizador BPE
python scripts/build_tokenizer_v1.py

# 3. Preparar dataset V1 en binario
python scripts/prepare_v1_corpus.py

# 4. Arrancar el pre-entrenamiento
python scripts/train.py --config configs/train_v1_high_density.yaml

# 5. Fine-tuning con LoRA
python scripts/finetune.py --lora_r 8 --lora_alpha 16 --data_file data/tool_dataset_real.json

# 6. (Opcional) Inferencia interactiva con tool-calling
python scripts/chat.py

# 7. Evaluar el modelo (perplexity y accuracy en tool-calling)
python scripts/eval.py --checkpoint checkpoints/ckpt_sft_latest.pt
```

## 📌 Nota sobre dataset
El finetuning por defecto usa `data/tool_dataset_real.json`, pero puedes cambiar el archivo con `--data_file`.

## 📦 Convención de checkpoints
- `ckpt_pretrain_latest.pt` y `ckpt_pretrain_best.pt` son checkpoints de pretraining.
- `ckpt_sft_latest.pt` es el checkpoint de fine-tuning.
- Los nombres antiguos siguen aceptados como fallback mientras se migra el repo.

---
*Construido artesanalmente desde cero.*
