# Plan de Entrenamiento V2: TinyThinker con Curriculum Learning

Basado en la evaluación del ciclo actual de entrenamiento (`logs/train_20260408_224354.log`), el modelo de 78M parámetros está logrando convergencia (bajando de loss 9.8 a ~1.5) pero sin garantía de estructuración lingüística o semántica pura. Dado que el coste computacional es alto en tu máquina (~410 segundos por cada 10 iteraciones en 8 threads de CPU), el próximo entrenamiento debe rentabilizar cada epoch usando **Curriculum Learning** para dominar gramática básica antes de ingerir hechos complejos.

## 1. El Problema Actual

Actualmente, el *dataloader* probablemente mezcla (o muestra aleatoriamente) `TinyStories` y `Wikipedia` o código. Al hacer esto:
* El modelo intenta aprender a sumar al mismo tiempo que intenta aprender qué es un número y un verbo.
* Esto causa picos bruscos de *loss* (como los vistos en las iteraciones 270, 1140, 1740) si, tras dominar una simple frase infantil, el modelo de repente se tropieza con un artículo complejo de Wikipedia sobre mecánica de fluidos.

## 2. Fase 1: Adquisición de Lenguaje (Gramática y Sintaxis)

**Objetivo:** Conseguir que el modelo genere frases en el idioma objetivo que suenen perfectas, aunque el contenido factual sea irrelevante (un "Orquestador Parlanchín").

* **Dataset:** 100% `TinyStories`.
* **Paradas (Thresholds):** Mantener esta dieta restrictiva hasta que la métrica de validación de *TinyStories* se estabilice profundamente y notemos que al probar el modelo, las oraciones tienen estructura sintáctica coherente (Sujeto + Verbo + Predicado).
* **Nota Técnica:** En el script `prepare_data.py`, debemos pre-tokenizar este set y guardarlo como `train_phase_1.bin`.

## 3. Fase 2: Introducción Guiada al Conocimiento

**Objetivo:** Sin dejar que el modelo olvide cómo hablar (evitar *olvido catastrófico* temprano), empezamos a inyectar información objetiva y vocabulario rico.

* **Dataset Híbrido:** 
  - 80% `TinyStories`
  - 20% `Wikipedia` (versión filtrada, donde los artículos no son masivos).
* **Por qué:** Mantiene viva la red sintáctica al tiempo que introduce los tokens que representan conceptos globales y nombres.
* **Transición en Código:** Modificar el `dataloader` para que abra dos archivos binarios (ej. pre-cache en `np.memmap`) y realice un muestreo de lotes (*batches*) basados en estas probabilidades.

## 4. Fase 3: Razonamiento y Data Sintética (El Córtex)

**Objetivo:** Enseñanza vertical. Forzarlo a establecer relaciones de alto nivel.

* **Dataset Híbrido Avanzado:**
  - 40% `Wikipedia`
  - 60% Datos Sintéticos Locales (Ejemplos lógicos, matemáticos, laberintos).
* **El Experimento Scratchpad:** Para probar las ideas de C.O.G.A, inyecta datos sintéticos aquí donde el modelo deba usar `<think>` y `<erase>` antes de emitir la conclusión. Con la base de TinyStories asentada, aprender y emitir etiquetas XML para simular cognición debería serle trivial.

## 5. Implementación en Código (Modificación Sugerida para el `dataloader`)

No necesitas frameworks complejos. Solo una pequeña lógica en tu generador de batches:

```python
# Ejemplo de lógica para el dataloader
def get_batch(split, current_iter):
    if current_iter < 5000:
        # FASE 1: Solo lenguaje base
        data = data_tinystories_train if split == 'train' else data_tinystories_val
    elif current_iter < 12000:
        # FASE 2: Lenguaje + Conocimiento (80/20)
        data = data_tinystories_train if random.random() < 0.8 else data_wikipedia_train
    else:
        # FASE 3: Lógica y C.O.G.A
        data = data_wikipedia_train if random.random() < 0.4 else data_synthetic_reasoning
    
    # ... código estándar para extraer el tensor de contexto y targets
```

## Resumen de Tareas para el próximo ciclo

1. Crear `prepare_data_phases.py` para tener archivos `.bin` separados por complejidad funcional.
2. Modificar la función `get_batch()` en el script de entrenamiento para inyectar este plan de fases.
3. Crear unos cientos de ejemplos usando etiquetas XML funcionales como `<think>` para enseñarle *pensamiento estructurado* a un LLM diminuto.
