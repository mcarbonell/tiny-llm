# 📚 SLM Research Notes (State of the Art)

> Documento vivo para recopilar ideas arquitectónicas y de entrenamiento extraídas de papers recientes de Modelos de Lenguaje Pequeños (SLMs) aplicables al proyecto TinyThinker (Especialmente Escala B: 50M Params).

## Analizados hasta el momento:
- `NVIDIA-Nemotron-3-Super-Technical-Report.pdf` (120B total, 12B activos - Abril 2026)
- `Nanbeige4-3B.pdf` (3B Params)

---

## 💡 1. Planificación del Learning Rate: FG-WSD (Nanbeige4)
El decaimiento típico del Learning Rate por coseno es subóptimo para modelos sometidos a currículums de datos.
* **Técnica:** Planificador Warmup-Stable-Decay (WSD).
* **Mecanismo:** Calentamiento rápido, mantener el LR fijo y alto durante el 80%-90% del entrenamiento (Fase Estable), y luego provocar una caída pronunciada (Fase Decay) al final. 
* **Aplicación a TinyThinker:** Implementar un scheduler WSD en `train.py` para darle mayor plasticidad al modelo en la Fase Estable y permitir caídas dramáticas en el Error Loss cuando empiece la Fase Decay.

## 💡 2. Data Curriculum por Fases (Nemotron & Nanbeige)
Mezclar todo el dataset uniformemente desdibuja el conocimiento de alta calidad.
* **Técnica:** Entrenamiento particionado.
* **Mecanismo:** 
  - Fase 1 (80% del tiempo): Texto genérico puro centrado en ganar vocabulario y estructura gramatical (Ej. TinyStories completo).
  - Fase 2 (20% final): Solo datos matemáticos, lógicos y conversacionales de máxima calidad (SimpleWiki, Tool-Calling). Se sincroniza esta fase con el `Decay` del Learning Rate.
* **Aplicación a TinyThinker:** Separar `train_combined.bin` y decirle a `train.py` que conmute de dataset en caliente (iteración ~16.000 para el `train_scale_b.yaml`).

## 💡 3. Destilación de Preferencias DPD (Nanbeige4)
Los modelos muy pequeños carecen del razonamiento inferencial profundo nativo de los Billones de parámetros.
* **Mecanismo:** Forzar la reestructuración del pensamiento pasando el output de modelos masivos por SFT en el modelo pequeño.
* **Estado en TinyThinker:** ✅ ¡Implementado! La generación de nuestro Dataset Dorado con LLMs profesores fuertes y los tags de Chain of Thought `<THINK>` resuelven esta arquitectura.

## 💡 4. Optimizaciones Críticas (Nemotron)
1. **Precisión FP4 (NVFP4):** Pre-entrenamientos híbridos directamente en 4 bits matemáticos. (Actualmente fuera de nuestro scope de CPU, pero brillante teóricamente).
2. **MoE y MTP (Multi-Token Prediction):** Aceleradores para inference throughput reduciendo los parámetros activos. 
* **Estado en TinyThinker:** Nosotros estamos mitigando el throughput con Grouped-Query Attention (GQA) con ratio 2:1 y Gradient Checkpointing en `train_scale_b.yaml`.

## 📂 5. Datasets Abiertos (NVIDIA Nemotron 3 Nano/Super)
Extraídos de los Technical Blogs de NVIDIA, son una mina de oro para nuestros futuros pre-entrenamientos y RL:
- **Pre-entrenamiento:** `https://huggingface.co/collections/nvidia/nemotron-pre-training-datasets` (10 Trillones de tokens curados, especial foco en razonamiento y código).
- **Post-Entrenamiento (SFT/RL):** `https://huggingface.co/collections/nvidia/nemotron-post-training-v3` (40M de muestras de alta calidad).
- **Entornos para Tool-Calling (RL):** `https://huggingface.co/collections/nvidia/nemo-gym` (Crucial para hacer a TinyThinker un experto en la terminal y navegador).

---
*Fecha de recolección: 8 de Abril de 2026.*
