# 🏭 The Cognition Factory: Estrategia de Dataset para Modelos $O(1)$

**Documento de Diseño: Planificación del "Master Corpus" futuro**
*Nota: Este documento describe la estrategia de ensamblaje para cuando el modelo esté maduro y se inicie el entrenamiento del "GPT-3 Homebrew".*

---

## 1. La Filosofía: Leyes > Hechos
Los modelos densos tradicionales ($O(d^2)$) tienen tanta capacidad bruta que pueden memorizar la Wikipedia entera a fuerza bruta. Un modelo Matrix-Free con Memoria O(1) está diseñado para ser un motor lógico, no una base de datos trivial.

Si entrenas un Hipocampo de Fourier puramente con artículos de Wikipedia ("La capital de Francia es París"), estarás desaprovechando su capacidad de seguir **estados y leyes**. Queremos que la red deduzca la física del lenguaje, no que se aprenda la enciclopedia.

## 2. La Receta del "Master Corpus V3"

Cuando llegue el momento de entrenar el modelo definitivo, la mezcla de datos (Mixture of Data) debería seguir esta proporción estricta:

### 🧩 40% - Pilar Lógico y Algorítmico (El Cerebro)
Datos generados sintéticamente que obligan a la red a razonar paso a paso, predecir estados futuros o deducir axiomas.
- **Fuentes:** Tus scripts `generate_planning_samples.py`, `generate_rich_logic_nim.py`, `generate_deep_poly_samples.py`.
- **Ejemplos:** Trazas de ejecución de Python, problemas de SAT, resolución de laberintos, partidas de ajedrez evaluadas.

### 🐘 20% - Pilar de Memoria Infinita (El Hipocampo)
Datos diseñados exclusivamente para estirar la memoria de retención del modelo. Contienen una dependencia causal entre el principio absoluto del documento y el final absoluto, separados por miles de tokens de distracción.
- **Fuentes:** `scripts/generate_hippocampus_stress_data.py` (Logs de servidores ficticios, dossiers de espionaje, código compilado con llaves de desencriptación remotas).
- **Objetivo:** Forzar a la red a no decaer el estado de las frecuencias bajas.

### 📚 40% - Pilar de Cultura Humana (El Lenguaje)
Datos humanos de alta calidad para que el modelo sepa expresarse con naturalidad, conozca el mundo real y respete la gramática.
- **Fuentes:** `fineweb_edu_10b.bin` u otros subsets de HuggingFace.
- **Objetivo:** Darle al motor lógico la capacidad de comunicarse fluida y coherentemente con el usuario.

## 3. Pipeline de Ensamblaje Futuro

Cuando decidas crear este dataset, el proceso será:
1. **Generación:** Ejecutar todos los scripts sintéticos para generar millones de archivos JSONL en `data/raw/`.
2. **Tokenización Unificada:** Usar el `tokenizer_v2_32k.json` para convertir todos esos textos en arrays binarios de enteros (`uint16`).
3. **Mezclado Aleatorio (Shuffling):** Usar un script en C o un script de Python optimizado (`mix_dataset.py`) para coger trozos (*chunks*) de los 3 pilares y mezclarlos. Si le das 1000 tokens de ajedrez seguidos de 1000 tokens de Wikipedia, evitas que la red sufra "olvido catastrófico" de una tarea mientras aprende otra.
4. **Validación:** Comprobar que el `.bin` final no tenga secuencias de padding infinitas y que los tokens clave del Hipocampo no se hayan truncado.

Este corpus será el combustible definitivo para las *Scaling Laws* del `RESEARCH_ROADMAP_V10`.
