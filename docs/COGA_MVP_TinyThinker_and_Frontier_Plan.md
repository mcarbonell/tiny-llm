# COGA en TinyThinker: MVP Implementable y Agenda de Frontera

## Propósito

Este documento propone una versión **implementable en este repo** de las ideas de COGA, sin tocar ni reescribir el documento original:

- [COGA-Cognitive-Operating-System-Architecture.md](C:\Users\mrcm_\Local\proj\tiny-llm\docs\COGA-Cognitive-Operating-System-Architecture.md)

La idea central es separar dos planos que en una discusión conceptual suelen mezclarse:

1. **Qué podemos construir aquí y ahora en TinyThinker**
2. **Qué haría un investigador de un gran lab con recursos de verdad**

El error más común en este tipo de arquitecturas es intentar implementar de golpe la versión "final". Aquí haremos lo contrario: elegir un MVP que preserve la tesis de COGA, pero que encaje con el estado real del repositorio.

---

## Resumen Ejecutivo

### Tesis práctica

El mejor MVP para este repo no es intentar construir COGA completo. El mejor MVP es demostrar tres cosas:

1. **Que el modelo puede usar memoria externa persistente**
2. **Que el modelo puede usar un espacio de trabajo externo para deliberación multi-paso**
3. **Que podemos medir si eso mejora tareas concretas**

### Qué no intentaremos en el MVP

- Scratchpad diferenciable intra-forward
- Coprocesadores inline dentro del paso de red
- Recurrencia adaptativa entrenada end-to-end
- Heartbeat autónomo fuerte
- Auto-finetuning nocturno

No porque sean malas ideas, sino porque en este repo todavía sería demasiado coste de complejidad para demasiado poco aprendizaje empírico.

### Propuesta de MVP

Construir un **COGA-lite runtime** alrededor del modelo actual o del modelo MoE:

- **Memoria persistente programable** fuera del modelo
- **Scratchpad externo estructurado** fuera del modelo
- **Bucle de deliberación controlado por runtime**
- **Presupuesto fijo de pasos**
- **Evaluación sobre tareas sintéticas y del repo**

En otras palabras: primero demostrar la arquitectura como sistema, y después, si funciona, mover piezas dentro de la red.

---

## Estado Actual del Repo

Este repositorio ya tiene varias bases útiles:

- Un modelo denso pequeño en [model.py](C:\Users\mrcm_\Local\proj\tiny-llm\model\model.py)
- Una variante MoE con expertos reservados en [model_moe.py](C:\Users\mrcm_\Local\proj\tiny-llm\model\model_moe.py)
- Chat con tool-calling interceptado en [chat.py](C:\Users\mrcm_\Local\proj\tiny-llm\scripts\chat.py)
- Evaluación básica en [eval.py](C:\Users\mrcm_\Local\proj\tiny-llm\scripts\eval.py)
- Tests de integración en [test_integration.py](C:\Users\mrcm_\Local\proj\tiny-llm\tests\test_integration.py)

Eso significa que no partimos de una idea vacía. Ya existe un embrión claro para:

- arquitectura base,
- runtime de interacción,
- tool use,
- MoE inicial,
- evaluación.

Por tanto, el MVP debería vivir sobre todo en `scripts/`, `tests/`, quizá un módulo nuevo en `model/` o `scratch/`, y datasets pequeños de control.

---

## Principio Rector del MVP

### No demostrar "inteligencia general"

El MVP no debe intentar responder:

- "¿COGA produce una mente artificial?"

Debe intentar responder preguntas mucho más estrechas:

- "¿La memoria externa mejora la coherencia entre sesiones?"
- "¿El scratchpad externo mejora tareas con backtracking?"
- "¿Un bucle de deliberación con presupuesto fijo mejora la tasa de acierto frente a una generación directa?"

Si no podemos responder eso con datos, hablar de heartbeat, sueño o auto-mejora es prematuro.

---

## MVP Propuesto: COGA-Lite Runtime

## Objetivo

Añadir al sistema actual un runtime que permita al modelo operar con estas primitivas:

- `remember(key, value, scope, importance)`
- `recall(query, top_k)`
- `scratch_write(slot, content)`
- `scratch_read(slot or query)`
- `scratch_edit(slot, content)`
- `scratch_delete(slot)`
- `final_answer(text)`

Estas primitivas no tienen por qué ser aprendidas dentro de la red en la primera iteración. En el MVP pueden ser ejecutadas por un controlador externo a partir de un formato textual estructurado.

### Ejemplo de protocolo

```text
<SCRATCH_WRITE slot="hyp1">
Si A > B y B > C, entonces A > C
</SCRATCH_WRITE>

<MEM_RECALL query="preferencias del usuario sobre Python" />

<FINAL_ANSWER>
La relación correcta es A > C.
</FINAL_ANSWER>
```

El runtime parsea estas emisiones, actualiza estado externo, y vuelve a llamar al modelo con el estado resumido.

Este diseño no es la versión final de COGA, pero sí conserva la hipótesis central:

- razonamiento desacoplado,
- memoria explícita,
- estado persistente,
- deliberación multi-paso.

---

## Componentes del MVP

## 1. Memoria Persistente Programable

### Qué es

Una base de memoria sencilla, persistida en disco, probablemente en JSONL o SQLite local, con embeddings opcionales más adelante.

### Qué debe soportar en v1

- inserción manual por el runtime,
- recall por coincidencia simple,
- filtrado por ámbito,
- TTL opcional,
- prioridad.

### Estructura mínima

```json
{
  "id": "mem_0001",
  "scope": "project",
  "kind": "episodic",
  "importance": "high",
  "source": "chat",
  "text": "El usuario prefiere respuestas concretas y orientadas a implementación.",
  "created_at": "2026-04-11T21:00:00Z",
  "last_accessed_at": "2026-04-11T21:05:00Z"
}
```

### Implementación recomendada

Crear un módulo nuevo, por ejemplo:

- `scratch/memory_store.py`

Con API tipo:

```python
store.remember(text, scope="project", kind="episodic", importance="medium")
store.recall(query, top_k=5, scope=None)
store.update(memory_id, text)
store.forget(memory_id)
```

### Por qué esto sí merece la pena

Porque es la pieza más barata y más medible de toda COGA. Además conecta muy bien con la tesis de "el modelo no debería empezar siempre desde cero".

---

## 2. Scratchpad Externo Estructurado

### Qué es

Una memoria de trabajo de corta duración, no persistente entre tareas, mantenida por el runtime.

### Qué debe soportar en v1

- slots nombrados,
- escritura, edición y borrado,
- resumen del estado actual,
- log de exploración.

### Estructura mínima

```json
{
  "task_id": "task_001",
  "slots": {
    "hyp1": "Hipótesis inicial",
    "calc1": "Caso base n=1"
  },
  "history": [
    "Created hyp1",
    "Edited hyp1",
    "Deleted branch_b"
  ]
}
```

### Implementación recomendada

Crear:

- `scratch/scratchpad.py`

Con operaciones:

```python
pad.write(slot, content)
pad.read(slot=None, query=None)
pad.edit(slot, content)
pad.delete(slot)
pad.summary()
pad.history_summary()
```

### Nota importante

En este MVP, el scratchpad no es tensorial ni diferenciable. Es una estructura de control externa. Eso es deliberado.

Primero hay que demostrar que:

- el modelo realmente lo usa,
- el uso correlaciona con mejora,
- el protocolo no se rompe.

Solo después tendría sentido moverlo "inside the network".

---

## 3. Deliberation Loop con Budget Fijo

### Qué es

Un bucle externo de N pasos donde el modelo puede:

- consultar memoria,
- manipular scratchpad,
- pedir herramienta,
- o responder.

### Algoritmo base

```text
1. Inyectar prompt + memorias relevantes + resumen de scratchpad
2. Pedir al modelo una acción estructurada
3. Ejecutar la acción en runtime
4. Actualizar estado
5. Repetir hasta:
   - FINAL_ANSWER
   - o budget agotado
```

### API recomendada

Crear un script nuevo, por ejemplo:

- `scripts/coga_chat.py`

Y quizás un módulo:

- `scripts/coga_runtime.py`

Con una clase:

```python
class COGARuntime:
    def run_task(self, user_prompt: str, max_steps: int = 6):
        ...
```

### Presupuesto inicial

Para el MVP, nada de predictor aprendido. Usar reglas:

- preguntas simples: `max_steps=2`
- razonamiento medio: `max_steps=4`
- tareas de búsqueda/backtracking: `max_steps=6`

Más tarde se puede añadir un clasificador heurístico o una pequeña head.

---

## 4. Tool Use como Submódulo del Runtime

El repo ya tiene una forma rudimentaria de tool-calling en [chat.py](C:\Users\mrcm_\Local\proj\tiny-llm\scripts\chat.py). El MVP debería reutilizar eso, no reinventarlo.

### Cambio conceptual

En lugar de:

- tool-calling como interrupción lineal del texto

pasar a:

- tool-calling como una acción más dentro del bucle de deliberación.

### Acciones mínimas

- `SEARCH(query)`
- `CALC(expression)` aunque sea mediante Python seguro local
- `LOOKUP_MEMORY(query)`

La combinación `scratchpad + memory + search` ya da un entorno muy rico para un MVP.

---

## 5. Evaluación que de verdad pruebe la tesis

Si no definimos tests adecuados, el MVP puede parecer impresionante sin demostrar nada.

### Benchmarks internos recomendados

#### A. Tareas con backtracking

Dataset sintético con problemas que requieran:

- probar una hipótesis,
- descartarla,
- intentar otra.

Ejemplos:

- mini acertijos lógicos,
- ordenaciones parciales,
- puzzles de restricciones simples,
- programación pequeña con depuración textual.

Métrica:

- exactitud final,
- número de pasos,
- tasa de corrección tras edición de scratchpad.

#### B. Tareas de memoria entre sesiones

Simular varias conversaciones donde el sistema deba recordar:

- preferencias del usuario,
- hechos del proyecto,
- decisiones previas.

Métrica:

- recall correcto,
- precisión de memorias recuperadas,
- tasa de contaminación por memorias irrelevantes.

#### C. Tareas con presupuesto

Comparar:

- generación directa,
- deliberación 2 pasos,
- deliberación 4 pasos,
- deliberación 6 pasos.

Métrica:

- accuracy,
- latencia,
- tokens generados,
- coste por acierto correcto.

### Archivos sugeridos

- `tests/test_coga_memory.py`
- `tests/test_coga_scratchpad.py`
- `tests/test_coga_runtime.py`
- `data/coga_mvp_eval.json`

---

## Qué sí implementaría yo en este repo

## Fase MVP-1: Runtime y estado externo

### Entregables

- `scratch/memory_store.py`
- `scratch/scratchpad.py`
- `scripts/coga_runtime.py`
- `scripts/coga_chat.py`

### Resultado esperado

Tener un agente que pueda:

- recordar cosas entre runs,
- mantener una working memory por tarea,
- operar en varios pasos,
- responder con formato final.

### Criterio de éxito

Que funcione end-to-end y pase tests deterministas sencillos.

---

## Fase MVP-2: Dataset y protocolo

### Entregables

- dataset pequeño con acciones estructuradas,
- parser robusto de acciones,
- prompts base del runtime,
- tests de errores de parseo.

### Idea importante

En este punto no hace falta que el modelo haya aprendido bien el protocolo. Podemos usar:

- few-shot prompting,
- plantillas,
- o incluso "teacher traces" sintéticas.

El objetivo es validar el circuito, no aún la elegancia del aprendizaje.

---

## Fase MVP-3: Evaluación comparativa

### Comparaciones mínimas

1. `baseline`: [chat.py](C:\Users\mrcm_\Local\proj\tiny-llm\scripts\chat.py) o generación simple
2. `memory_only`
3. `scratchpad_only`
4. `memory + scratchpad + budgeted loop`

### Resultado que justificaría seguir

Seguiría invirtiendo si vemos al menos una de estas señales:

- mejora clara en tareas multi-paso,
- mejora clara en continuidad entre sesiones,
- mejor tasa de autocorrección antes de responder.

Si no aparece ninguna, entonces la tesis no está validada en esta escala.

---

## Qué NO haría todavía en este repo

## 1. Scratchpad diferenciable dentro del forward

Esto suena tentador, pero aquí sería un error prematuro.

Riesgos:

- complejidad brutal de entrenamiento,
- difícil depurar,
- fácil confundir inestabilidad con "nueva capacidad",
- coste alto para un modelo pequeño.

Primero hay que aprender qué operaciones de scratchpad son realmente útiles.

## 2. Heartbeat autónomo real

En un MVP local, un heartbeat activo añade ruido, complejidad de producto y riesgo de falsas alarmas.

Lo máximo razonable sería un script batch tipo:

- `scripts/coga_review_memory.py`

que revise logs bajo demanda.

## 3. Auto-finetuning nocturno

Todavía no. Antes de auto-modificarse, el sistema debe:

- medir sus fallos,
- tener datasets fiables,
- tener un rollback claro,
- demostrar que no se degrada.

## 4. Coprocesadores inline

En este repo hay que pensarlos como herramientas del runtime, no como operaciones dentro del forward pass.

---

## Qué haría un investigador de un gran lab

Aquí cambiamos completamente de escala.

Un gran lab no debería empezar por el "wrapper externo" como producto final. Lo usaría solo como sonda experimental. Su objetivo real sería identificar qué piezas merecen internalizarse en la arquitectura.

## Programa de investigación de frontera

## Línea 1. Scratchpad differentiable pero restringido

No intentaría una RAM universal desde el día 1. Haría algo intermedio:

- un banco pequeño de slots latentes,
- operaciones limitadas,
- write/read gates,
- coste explícito por escribir y reescribir.

### Preguntas científicas

- ¿El modelo aprende a usar scratch slots de forma no trivial?
- ¿Aparece autocorrección real?
- ¿Qué tareas disparan más escrituras útiles?
- ¿Compensa frente a simplemente dar más tokens de pensamiento?

### Experimentos

- comparar CoT largo vs scratchpad latente de tamaño fijo,
- medir mejora por unidad de compute,
- medir tasa de revisiones internas antes de respuesta final.

## Línea 2. Recurrencia adaptativa de verdad

Un lab con recursos sí debería probar un bloque recurrente serio sobre transformer o SSM+transformer híbrido.

### Lo interesante aquí

No es solo "más pasos". Es:

- halting,
- reutilización de pesos,
- profundidad variable condicionada por dificultad,
- training estable.

### Métricas clave

- accuracy vs FLOPs,
- distribución de pasos por tarea,
- calibración del halting,
- generalización a problemas fuera de distribución.

## Línea 3. Memoria a largo plazo con write policy aprendida

La frontera no está en hacer RAG clásico. Está en aprender:

- cuándo guardar,
- qué resumir,
- cuándo olvidar,
- cómo evitar memoria tóxica o redundante.

### Programa razonable

- empezar con heurísticas,
- luego entrenar policy de escritura,
- después introducir consolidación offline,
- finalmente medir utilidad causal de las memorias.

La palabra clave aquí es **causal**. No basta con que haya memorias; hay que demostrar que ayudan.

## Línea 4. Routing modular y expert slots de verdad

El repo ya apunta a esto en [model_moe.py](C:\Users\mrcm_\Local\proj\tiny-llm\model\model_moe.py), pero un gran lab haría el estudio serio.

### Preguntas reales

- ¿Los slots reservados evitan degradación real o solo la desplazan al router?
- ¿Qué granularidad de routing funciona mejor, token, span o secuencia?
- ¿Cómo se componen expertos nuevos sin destruir calibración?
- ¿Se puede versionar un experto como artefacto transferible entre modelos cercanos?

### Lo que haría

- ablation por tamaño de slot,
- frozen base + trainable router,
- frozen base + partially trainable router,
- benchmark de retención general antes y después.

## Línea 5. Herramientas integradas en activaciones

Esta es probablemente la línea más arriesgada y más "big lab only".

No me creería la versión fuerte de "coprocesador inline sin latencia" hasta demostrar un sistema híbrido convincente.

### Lo mínimo serio

- dispatch differentiable o semidifferentiable,
- tipos de herramienta muy restringidos al principio,
- latencia medida,
- fallback robusto.

Yo empezaría con:

- calculadora exacta,
- verificador simbólico limitado,
- lookup sobre memoria interna/externa.

No empezaría por ejecución arbitraria de código dentro del bucle principal.

## Línea 6. Ciclo nocturno con gobernanza fuerte

Si un gran lab explora auto-mejora, debería tratarlo como un pipeline de entrenamiento seguro, no como magia emergente.

### Componentes obligatorios

- logging estructurado,
- selección conservadora de ejemplos,
- validación offline,
- canary deployment,
- rollback automático,
- auditoría de qué cambió y por qué.

La frontera no es "que el modelo se mejore solo". La frontera es "que lo haga sin degradarse, sin alucinar mejoras y sin retroalimentar sesgos malos".

---

## Mi priorización si yo estuviera en un gran lab

Invertiría en este orden:

1. **Recurrencia adaptativa**
2. **Scratchpad latente pequeño y medible**
3. **Memory write policy**
4. **Expert slots / modularidad**
5. **Tool integration**
6. **Auto-improvement nocturno**

La razón es simple:

- recurrencia y scratchpad atacan el núcleo de la deliberación,
- memoria y modularidad atacan continuidad y especialización,
- herramientas y auto-mejora tienen mucho valor, pero también más riesgo de convertir el sistema en una colección de hacks mal acoplados.

---

## Roadmap Concreto para Este Repo

## Etapa 1. Infraestructura COGA-lite

### Objetivo

Tener runtime, memoria y scratchpad funcionando.

### Trabajo

- crear `memory_store.py`
- crear `scratchpad.py`
- crear `coga_runtime.py`
- crear `coga_chat.py`
- tests unitarios

### Tiempo estimado

Pocos días de trabajo si mantenemos alcance estricto.

---

## Etapa 2. Dataset de evaluación COGA

### Objetivo

No depender de impresiones subjetivas.

### Trabajo

- diseñar 50-200 ejemplos pequeños,
- dividir por tipo de capacidad,
- añadir baseline y comparación.

### Tiempo estimado

Uno o dos días para una primera versión útil.

---

## Etapa 3. Integración con MoE

### Objetivo

Conectar el runtime con la línea de modularidad del repo.

### Trabajo

- definir si el runtime usa `TinyThinker` o `TinyThinkerMoE`,
- preparar evaluación de expertos reservados,
- no mezclar todavía eso con memoria persistente aprendida.

### Riesgo

Si se hace demasiado pronto, será difícil saber qué mejora viene de qué.

---

## Etapa 4. Deliberación entrenada

### Objetivo

Pasar de protocolo externo "guiado" a uso más natural por el modelo.

### Trabajo

- traces sintéticas,
- SFT del protocolo de acciones,
- quizá RL ligero después.

### Condición previa

Que la etapa 1 y 2 ya hayan mostrado alguna señal de valor real.

---

## Criterios de éxito y fracaso

## Éxito

Consideraría este MVP un éxito si demuestra al menos dos de estas tres cosas:

1. Mejora medible en tareas con backtracking o exploración
2. Mejora medible en continuidad entre sesiones
3. Mejora medible en autocorrección antes de responder

## Fracaso útil

También sería un buen resultado descubrir que:

- el scratchpad externo apenas ayuda,
- la memoria contamina más de lo que aporta,
- o el loop multi-step solo añade coste sin mejorar acierto.

Eso nos diría que la tesis necesita rediseño antes de mover nada a nivel arquitectural.

---

## Conclusión

La forma correcta de llevar COGA a este repo no es intentar construir una mente artificial completa. Es construir un **sistema experimental disciplinado** que permita validar las partes más prometedoras.

Mi recomendación es:

1. **Implementar ya** memoria persistente, scratchpad externo y runtime multi-paso
2. **Medir ya** si ayudan de verdad
3. **Posponer** la internalización arquitectural hasta tener evidencia
4. **Reservar** las ideas más ambiciosas para una agenda de investigación de frontera, no para el MVP

La intuición de fondo de COGA me parece buena. Pero la mejor manera de respetarla aquí es no romantizarla: hay que convertirla en hipótesis falsables y en componentes que podamos encender, medir y comparar.

---

## Anexo: Primer corte de archivos a crear

```text
docs/
  COGA_MVP_TinyThinker_and_Frontier_Plan.md

scratch/
  memory_store.py
  scratchpad.py

scripts/
  coga_runtime.py
  coga_chat.py

tests/
  test_coga_memory.py
  test_coga_scratchpad.py
  test_coga_runtime.py
```

## Anexo: Primera pregunta experimental

Si solo pudiéramos probar una hipótesis en este repo, yo elegiría esta:

**¿Un loop de deliberación con scratchpad externo fijo mejora el rendimiento frente a generación directa en tareas de razonamiento pequeño pero con necesidad de corrección intermedia?**

Si la respuesta es sí, merece la pena seguir.
Si la respuesta es no, conviene revisar la tesis antes de complejizar el sistema.

---

Doc by GPT 5.4