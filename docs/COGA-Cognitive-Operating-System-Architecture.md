# COGA: Cognitive Operating System Architecture
## Del LLM al Agente Cognitivo Persistente

### Un manifiesto para la próxima generación de inteligencia artificial

---

*Documento generado a partir de un brainstorming colaborativo entre un humano y una IA. Las ideas aquí expuestas son producto de una conversación genuina donde ambas partes contribuyeron intuiciones, analogías y diseños técnicos. Quizás eso mismo ya dice algo sobre el futuro que aquí se describe.*

---

## Prólogo

Este documento nace de una pregunta aparentemente simple: **¿cómo conseguir LLMs más inteligentes con menos parámetros y compute?** Lo que comenzó como una exploración técnica sobre eficiencia se transformó progresivamente en algo más ambicioso: el diseño de una arquitectura cognitiva completa que trasciende lo que hoy entendemos por "modelo de lenguaje".

El argumento central es que los LLMs actuales no están limitados por tamaño sino por **arquitectura cognitiva**. Son un cortex sin hipocampo, un procesador sin RAM editable, un pensador que no puede pensar sin hablar. Las piezas que les faltan no son misteriosas ni requieren avances teóricos fundamentales — son identificables, diseñables e implementables con tecnología actual o de corto plazo.

El documento se estructura en tres partes:

- **Parte I**: Diagnóstico — qué le falta a los LLMs actuales
- **Parte II**: Arquitectura COGA — diseño integrado de la solución
- **Parte III**: Visión — implicaciones y futuro

---

# PARTE I: DIAGNÓSTICO

## 1. El estado actual: potencia bruta, arquitectura pobre

Los LLMs actuales representan un logro extraordinario y una ineficiencia extraordinaria al mismo tiempo. GPT-4, con sus estimados ~1.8 trillones de parámetros, consume recursos comparables a una pequeña central eléctrica durante su entrenamiento. El cerebro humano, con ~86 mil millones de neuronas y ~20W de consumo — menos que una bombilla — supera a estos modelos en razonamiento general, eficiencia, aprendizaje continuo y adaptación.

La diferencia no está en la escala sino en la organización.

### 1.1 Lo que los LLMs hacen extraordinariamente bien

- Reconocimiento de patrones en lenguaje a escala masiva
- Compresión y síntesis de conocimiento
- Transferencia entre dominios
- Seguimiento de instrucciones complejas
- Generación de texto coherente y contextual

### 1.2 Lo que les falta: un inventario de carencias

```
CARENCIA 1: Memoria de trabajo editable
    Estado actual: la "memoria" es el contexto, y es append-only.
    Cada token emitido es permanente e irrevocable.
    No pueden explorar y descartar. No pueden pensar sin hablar.
    
CARENCIA 2: Memoria a largo plazo programable
    Estado actual: todo el conocimiento está en los pesos (fijo)
    o en el contexto (temporal y volátil).
    No pueden aprender de la experiencia entre conversaciones.
    No pueden decidir qué recordar y qué olvidar.

CARENCIA 3: Computación adaptativa
    Estado actual: gastan el mismo compute en "2+2" que en 
    "demuestra la conjetura de Goldbach". Cada token pasa 
    por todas las capas, todos los parámetros relevantes, 
    sin importar la dificultad.

CARENCIA 4: Herramientas integradas
    Estado actual: para usar una calculadora, el modelo debe 
    dejar de pensar, emitir una llamada, esperar, recibir 
    resultado, y reiniciar. Es como si un humano tuviera que 
    dejar de pensar para usar sus dedos.

CARENCIA 5: Modularidad real
    Estado actual: adquirir conocimiento nuevo (fine-tuning) 
    destruye conocimiento anterior (olvido catastrófico).
    No pueden especializarse sin sacrificar generalidad.

CARENCIA 6: Continuidad temporal
    Estado actual: los LLMs no existen entre requests.
    No tienen heartbeat. No pueden reflexionar, planificar 
    o mejorar fuera de una conversación activa.

CARENCIA 7: Recurrencia
    Estado actual: un transformer de L capas tiene exactamente 
    L pasos de razonamiento. Si el problema requiere L+1, falla.
    No puede "seguir pensando" un poco más.
```

### 1.3 La metáfora del cortex aislado

La mejor forma de entender un LLM actual es como un **cortex cerebral aislado** — la parte del cerebro responsable del razonamiento de alto nivel, separada de todas las estructuras que lo apoyan:

```
CEREBRO HUMANO COMPLETO:
    Cortex prefrontal  → Razonamiento, planificación
    Hipocampo           → Memoria episódica, aprendizaje
    Cortex sensorial    → Percepción
    Cerebelo            → Automatización de procedimientos
    Amígdala            → Evaluación emocional/relevancia
    Tálamo              → Routing de información
    Sistema reticular   → Atención, arousal, regulación
    Ganglios basales    → Selección de acciones
    
LLM ACTUAL:
    Cortex (parcial)    → Razonamiento, reconocimiento de patrones
    ... y nada más.
```

Un cortex aislado puede hacer cosas impresionantes — reconocer patrones, generar lenguaje, razonar hasta cierto punto. Pero sin las estructuras de soporte, está fundamentalmente limitado. No puede formar nuevos recuerdos. No puede regular su propia atención. No puede automatizar procedimientos aprendidos. No puede mantener continuidad temporal.

COGA propone **reconstruir las estructuras faltantes**.

---

# PARTE II: ARQUITECTURA COGA

## 2. Visión general

COGA (Cognitive Operating System Architecture) trata al modelo de lenguaje no como un producto final sino como el **núcleo razonador** de un sistema cognitivo más amplio. Alrededor de ese núcleo, construye las estructuras necesarias para memoria, aprendizaje, adaptación y continuidad.

La analogía operativa es un **sistema operativo**:

```
┌───────────────────────────────────────────────────────────┐
│                    COGA: MAPA COMPLETO                    │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              HEARTBEAT (continuidad temporal)       │  │
│  │  Tick periódico que mantiene al agente "vivo"       │  │
│  │  Frecuencia adaptativa según actividad              │  │
│  └──────────────────────┬──────────────────────────────┘  │
│                         │                                 │
│  ┌──────────────────────▼──────────────────────────────┐  │
│  │           CONTROLADOR DE RECURSOS                   │  │
│  │  Budget estimator + scheduler + prioridades         │  │
│  └──────────────────────┬──────────────────────────────┘  │
│                         │                                 │
│  ┌──────────────────────▼──────────────────────────────┐  │
│  │              NÚCLEO RAZONADOR                       │  │
│  │  Transformer heterogéneo + MoE + recurrencia        │  │
│  │  El "cortex" — razonamiento puro                    │  │
│  │                                                     │  │
│  │  ┌─────────┐ ┌────────────┐ ┌──────────────────┐    │  │
│  │  │Bloque A │→│  Bloque B  │→│    Bloque C      │    │  │
│  │  │ (SSM)   │ │(Transformer│ │   (Output)       │    │  │
│  │  │ Parsing │ │ +MoE       │ │  Refinamiento    │    │  │
│  │  └─────────┘ │ recurrente)│ └──────────────────┘    │  │
│  │              │     ↺      │                         │  │
│  │              └────────────┘                         │  │
│  └──────────┬──────────┬──────────┬────────────────────┘  │
│             │          │          │                       │
│  ┌──────────▼───┐ ┌────▼─────┐ ┌──▼────────────────┐      │
│  │ SCRATCHPAD   │ │ MEMORIAS │ │ COPROCESADORES    │      │
│  │ MUTABLE      │ │          │ │                   │      │
│  │ Working mem. │ │ remember │ │ calc()            │      │
│  │ + log de     │ │ recall   │ │ lookup()          │      │
│  │ exploración  │ │ forget   │ │ verify()          │      │
│  │              │ │ update   │ │ execute_code()    │      │
│  └──────────────┘ └────┬─────┘ └───────────────────┘      │
│                        │                                  │
│  ┌─────────────────────▼───────────────────────────────┐  │
│  │              EXPERT SLOTS (MES)                     │  │
│  │  Base experts (frozen) + slots modulares            │  │
│  │  Plug-and-play domain knowledge                     │  │
│  └─────────────────────┬───────────────────────────────┘  │
│                        │                                  │
│  ┌─────────────────────▼───────────────────────────────┐  │
│  │           CICLO NOCTURNO (sueño artificial)         │  │
│  │  Análisis de logs → detección de patrones →         │  │
│  │  auto-prescripción → fine-tune → verificación       │  │
│  │  Consolidación de memorias episódicas → pesos       │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

### Tabla de equivalencias

| Componente COGA | Análogo cerebral | Análogo computacional | Función |
|---|---|---|---|
| Núcleo razonador | Cortex prefrontal | CPU | Razonamiento central |
| Scratchpad mutable | Memoria de trabajo | RAM (read/write) | Pensamiento exploratorio |
| Log de exploración | Experiencia reciente | Write-ahead log | Evitar bucles |
| Memorias programables | Hipocampo | Base de datos | Aprendizaje entre sesiones |
| Expert Slots | Áreas corticales especializadas | Microservicios | Conocimiento modular |
| Coprocesadores | Cerebelo, áreas motoras | FPU, coprocesadores | Herramientas integradas |
| Budget estimator | Sistema reticular activador | Scheduler del OS | Regulación de recursos |
| Heartbeat | Ritmos circadianos, arousal | Cron + event loop | Continuidad temporal |
| Ciclo nocturno | Sueño (consolidación) | Batch job nocturno | Auto-mejora |
| Router | Tálamo | Load balancer | Routing de información |

---

## 3. Componente 1: Núcleo Razonador Heterogéneo

### 3.1 El problema de la homogeneidad

Los transformers actuales usan capas idénticas repetidas — misma dimensión, mismo número de heads, mismo tamaño de FFN. Esto es por conveniencia ingenieril, no por optimalidad. La investigación sobre pruning muestra que diferentes capas tienen importancia muy diferente:

- **Capas iniciales**: procesan features sintácticos y superficiales
- **Capas intermedias**: realizan el razonamiento semántico profundo
- **Capas finales**: muchas son casi redundantes, refinan formatos

### 3.2 Diseño de bloques funcionales

COGA divide el núcleo en tres bloques con roles diferenciados:

**Bloque A — Parsing y Representación (SSM, pequeño)**
```
Tipo:       State Space Model (Mamba-2 style)
Dimensión:  d_model = 2048 (pequeño)
Capas:      4
Función:    procesamiento secuencial eficiente, 
            construcción de representaciones básicas
Recurrente: No (se ejecuta una sola vez)
```

Las primeras capas de un transformer procesan features que no requieren atención cuadrática completa. Un SSM (complejidad lineal) es más eficiente para esta fase.

**Bloque B — Razonamiento Profundo (Transformer + MoE, grande, recurrente)**
```
Tipo:       Transformer con atención densa + MoE
Dimensión:  d_model = 6144 (grande)
Capas:      12
MoE:        16 expertos por capa (12 base + 4 slots), top-2 routing
Conexiones: cross-attention a scratchpad, memoria, coprocesadores
RECURRENTE: este bloque puede ejecutarse N veces
```

El bloque central concentra la mayoría de la capacidad computacional donde más importa — en el razonamiento profundo. Su recurrencia permite profundidad de pensamiento variable.

**Bloque C — Refinamiento y Output (Transformer, medio)**
```
Tipo:       Transformer estándar
Dimensión:  d_model = 4096 (medio)
Capas:      6
Función:    convertir representaciones internas en output
Recurrente: No (se ejecuta una sola vez)
```

### 3.3 Recurrencia adaptativa

La recurrencia del Bloque B es el mecanismo que permite al modelo **ajustar la profundidad de razonamiento** al problema:

```
Sin recurrencia:  Profundidad FIJA de L capas.
                  Si el problema necesita L+1 pasos → falla.
                  Clase computacional: circuitos de profundidad fija (TC⁰)

Con recurrencia:  Profundidad VARIABLE.
                  El modelo itera hasta que "sabe" la respuesta.
                  Clase computacional: Turing-completa (dado tiempo suficiente)
```

Un modelo de 12 capas sin recurrencia tiene exactamente 12 pasos de razonamiento. El mismo modelo con recurrencia x8 tiene 96 pasos de razonamiento potenciales — **con los mismos parámetros**. Esto es más compute, no más parámetros.

El mecanismo de halting combina cuatro safety nets:

1. **Budget aprendido**: el modelo estima la dificultad antes de empezar y se asigna un presupuesto
2. **Halt probability**: cada iteración produce p(parar), acumulada hasta un threshold
3. **Detección de convergencia**: si el estado oculto deja de cambiar significativamente, se para
4. **Hard cap absoluto**: nunca más de N iteraciones (safety net final)

El loss de entrenamiento es dual:

```
L = L_accuracy + λ · L_compute
```

Esto incentiva al modelo a ser **correcto y eficiente**: pensar lo suficiente pero no más de lo necesario. Exactamente como un jugador de ajedrez con reloj.

### 3.4 Parámetros estimados

```
Bloque A (SSM):                ~0.5B parámetros
Bloque B (Transformer + MoE):  ~40B parámetros totales, ~5B activos
Bloque C (Transformer):        ~2B parámetros

TOTAL:                         ~45B parámetros totales
ACTIVOS por forward pass:      ~8-10B parámetros
Con recurrencia x4:            compute equivalente a ~35B denso
                               pero con profundidad de razonamiento
                               potencialmente ilimitada
```

---

## 4. Componente 2: Scratchpad Mutable — Pensar sin hablar

### 4.1 El problema fundamental

Los LLMs actuales tienen una propiedad que, examinada cuidadosamente, es absurda: **pensar y comunicar son la misma operación**. Cada token generado es simultáneamente un paso de computación interna y una emisión pública. Es como si un humano no pudiera pensar sin hablar en voz alta.

Las consecuencias son severas:

- **Sin exploración**: no pueden considerar 5 hipótesis y elegir la mejor
- **Compromiso prematuro**: cada token condiciona irreversiblemente los siguientes
- **Sin autocorrección**: si se equivocan en el paso 3 de 20, arrastran el error
- **Consumo de contexto**: tokens de "pensamiento" consumen ventana de contexto permanentemente

El chain-of-thought (incluyendo los "thinking tokens" de modelos como o1 y R1) mitiga pero no resuelve: sigue siendo append-only, lineal, e irrevocable.

### 4.2 La analogía del ajedrez

La analogía más precisa viene del ajedrez:

```
Gran maestro humano:
    Piensa: "Si muevo alfil a e5... no, me come la torre"
    Piensa: "Si muevo caballo a f3... amenaza mate"
    Piensa: "Pero entonces él puede hacer Dc7..."
    Piensa: "OK, caballo f3 es la mejor tras analizar 12 líneas"
    HABLA: "Caballo f3"
    
    → 12 líneas exploradas, 11 descartadas, 1 emitida
    → El "tablero mental" se borró y reescribió muchas veces
    → Las líneas descartadas no "pesan" en la decisión final

Motor de ajedrez (Stockfish):
    make_move(e5) → evaluate() → undo_move()
    make_move(f3) → evaluate() → undo_move()
    make_move(d4) → evaluate() → undo_move()
    → best_move = f3
    
    → Explícitamente hace/deshace/evalúa/descarta
    → La memoria (posición) es mutable

LLM actual:
    Emite: "Voy a mover el alfil a e5 porque..."
    → Ya se comprometió. No hay undo. No hay alternativas.
    → Si e5 era malo, toda la cadena posterior es incorrecta.
```

### 4.3 Diseño del scratchpad

El scratchpad es un **banco de memoria key-value externo** al contexto principal, con operaciones CRUD completas:

```
OPERACIONES:
    WRITE(content)              → Escribe una hipótesis/dato en un slot
    READ(query)                 → Recupera por similitud semántica
    EDIT(slot_id, new_content)  → Modifica contenido existente
    DELETE(slot_id)             → Libera un slot
    COMPARE(slot_a, slot_b)     → Compara dos hipótesis
    COMMIT(slot_id)             → Marca como "respuesta final"
    NOP                         → No hacer nada con el scratchpad

ESTRUCTURA:
    Slots: 32 (configurable)
    Dimensión por slot: d_model (6144)
    Cada slot: key + value + status + version + confidence
    
INTERACCIÓN:
    Cross-attention bidireccional con el Bloque B
    El modelo lee y escribe en el scratchpad en cada capa
    del Bloque B, en cada iteración de recurrencia
```

### 4.4 El insight clave: la ventana de contexto como RAM editable

La ventana de contexto de un LLM actual es **memoria de solo append**. El scratchpad la transforma en **RAM editable**. La diferencia computacional es fundamental:

```
Append-only:   Clase de algoritmos implementables LIMITADA
               No puede implementar backtracking
               No puede implementar búsqueda con poda
               No puede implementar iteración con estado mutable

RAM editable:  TURING-COMPLETA
               Puede implementar CUALQUIER algoritmo
               Backtracking, búsqueda, poda, iteración, recursión
```

Esto permite algo revolucionario: **pensamiento extendido sin crecimiento de contexto**.

```
Modelo actual con CoT:
    Input: 1,000 tokens
    Pensamiento (CoT): 100,000 tokens (append-only, crece linealmente)
    Output: 500 tokens
    Ventana consumida: 101,500 tokens
    → Limitado por tamaño de ventana
    → Atención cuadrática crece prohibitivamente
    → Calidad se degrada en contextos muy largos

Modelo con scratchpad mutable:
    Input: 1,000 tokens
    Scratchpad: 32 slots × 512 tokens = 16,384 tokens (tamaño FIJO)
    Pero los slots se REESCRIBEN miles de veces durante el pensamiento
    → Procesamiento equivalente a millones de tokens
    → Sin crecer la ventana de contexto
    → Atención tiene coste constante (scratchpad es fijo)
    Output: 500 tokens
    Ventana real: ~18,000 tokens (constante)
    Pensamiento efectivo: ilimitado
```

Es exactamente como funciona una CPU con RAM. La RAM tiene tamaño fijo — 32GB — pero un programa puede ejecutarse durante horas, leyendo y reescribiendo esa misma RAM millones de veces. El tamaño de la RAM no limita la duración del cómputo.

### 4.5 Prevención de bucles: el log de exploración

Un problema potencial: al borrar hipótesis descartadas del scratchpad, el modelo pierde el registro de qué ya intentó y podría entrar en bucles.

La solución es un **sistema de dos niveles**:

```
NIVEL 1: SCRATCHPAD ACTIVO (mutable)
    Lo que el modelo está pensando AHORA
    Totalmente editable
    Se borra, se reescribe, se reorganiza

NIVEL 2: LOG DE EXPLORACIÓN (append-only)
    Cuando algo se descarta del scratchpad activo,
    se escribe un RESUMEN comprimido en el log
    "Hipótesis A descartada, razón: inconsistente con dato X"
    "Vía B explorada parcialmente, resultado: sin convergencia"
    
    El modelo consulta el log antes de explorar nuevas vías
    → Evita redescubrir caminos ya descartados
```

Análogo en ajedrez: tu cálculo activo es mutable (mueves piezas mentalmente), pero mantienes una sensación de "esta línea ya la miré y no me gustó". En Stockfish es la **transposition table** — no recalcula posiciones ya evaluadas.

### 4.6 Entrenamiento del scratchpad

El entrenamiento sigue un protocolo de cuatro fases:

**Fase 1 — Dataset sintético**: un modelo frontera genera traces simulando uso de scratchpad en formato estructurado. Estas traces son imperfectas pero suficientes para bootstrap.

**Fase 2 — Supervised Fine-Tuning**: el modelo aprende la mecánica del scratchpad — cuándo escribir, leer, editar, borrar. No necesita ser óptimo, solo funcional.

**Fase 3 — Reinforcement Learning**: el modelo aprende cuándo el scratchpad realmente ayuda. El reward premia respuestas correctas y penaliza uso innecesario del scratchpad. Premia especialmente la autocorrección (editar o borrar una hipótesis incorrecta).

**Fase 4 — Self-play**: el modelo genera sus propios datos de entrenamiento. Resuelve problemas, filtra por calidad, re-entrena con los mejores ejemplos. El uso del scratchpad mejora progresivamente.

### 4.7 Impacto estimado en capacidades

```
Razonamiento matemático (MATH, AIME):     +20-40%
    El modelo puede explorar enfoques, descartar, backtrack
    
Programación compleja (SWE-bench):         +15-30%
    Puede diseñar, simular, encontrar bugs, rediseñar
    
Planificación multi-paso:                  +25-50%
    Puede explorar árboles de decisión, podar ramas
    
Problemas creativos/abiertos:              Cualitativamente transformador
    Pensamiento exploratorio libre sin ansiedad de comprometerse
    
QA factual:                                +5-10%
    Verificación interna antes de responder
    
Tareas simples/directas:                   ~0%
    No necesitan exploración
```

El patrón es claro: **cuanto más se beneficia la tarea de exploración y backtracking, más impacto tiene el scratchpad**. Y precisamente esas son las tareas donde los LLMs actuales fallan más.

---

## 5. Componente 3: Memorias Programables

### 5.1 De la amnesia a la experiencia

Un LLM actual tiene **amnesia anterógrada total**: no puede formar nuevos recuerdos a largo plazo. Cada conversación empieza desde cero. No puede aprender de sus errores. No puede adaptarse a un usuario. No puede acumular experiencia.

COGA introduce un sistema de memoria programable con primitivas que el propio modelo invoca:

```python
# Primitivas de memoria

remember(context, info)
    # El modelo decide: "esto es importante, quiero recordarlo"
    # Ejemplo: remember(
    #     "cuando el usuario pida código Python",
    #     "prefiere type hints, docstrings, y pytest"
    # )

recall(query) → memories
    # El modelo (o el motor de inferencia) busca memorias relevantes
    # Búsqueda por: similitud semántica, keywords, contexto, ámbito

forget(memory_id)
    # El modelo decide que un recuerdo ya no es necesario
    # Ejemplo: forget("mem_0847")  # usuario cambió de framework

update_memory(memory_id, new_info)
    # El modelo actualiza un recuerdo existente
    # Ejemplo: update_memory("mem_0231", 
    #     "el usuario ahora prefiere respuestas concisas")
```

### 5.2 Tipos de memoria

```
MEMORIA SEMÁNTICA (hechos, conocimiento)
    "La API de Stripe usa idempotency keys para evitar duplicados"
    Persiste: indefinidamente
    Actualización: cuando el modelo detecta información nueva/corregida

MEMORIA EPISÓDICA (experiencias, contexto)  
    "En la conversación del 15/03, el usuario y yo diseñamos 
     una arquitectura de microservicios para su proyecto"
    Persiste: configurable (sesión, proyecto, permanente)
    Actualización: automática tras cada interacción

MEMORIA PROCEDURAL (cómo hacer cosas)
    "Para resolver integrales por partes: 1) identificar u y dv 
     usando LIATE, 2) verificar derivando, 3) si no sale, 
     probar sustitución primero"
    Persiste: indefinidamente
    Actualización: cuando el modelo encuentra un procedimiento mejor
```

### 5.3 Inyección de memorias durante inferencia

Las memorias se inyectan automáticamente al inicio de cada inferencia, seleccionadas por relevancia:

```xml
<memories>
  <memory id="mem_0124" type="episodic" relevance="0.94">
    Cuando el usuario pida código Python:
    - Usar type hints y docstrings
    - Usar pytest para tests
    - Estilo funcional preferido sobre OOP
  </memory>
  <memory id="mem_0892" type="semantic" relevance="0.71">
    El usuario trabaja con Python 3.11+
    Puede usar match/case y tomllib
  </memory>
</memories>

<user>
  Hazme una función que ordene una lista
</user>
```

En versiones avanzadas, la inyección ocurriría **mid-inference**: activaciones intermedias del Bloque B dispararían búsquedas asociativas en el banco de memorias, con resultados inyectados via cross-attention. Esto replica el mecanismo del hipocampo: pattern completion automático, no búsqueda consciente.

### 5.4 El modelo como auto-programador

Lo más profundo de `remember()` es que el modelo **se programa a sí mismo**. No solo recuerda hechos — se da instrucciones para su comportamiento futuro:

```python
# El modelo aprendiendo de un error:
remember(
    "cuando me pregunten sobre población de países",
    "verificar con coprocesador de lookup antes de responder. 
     La última vez dije 35M para Canadá y son ~40M. 
     No confiar en datos demográficos de memoria."
)

# El modelo optimizando su propio razonamiento:
remember(
    "problemas de combinatoria",
    "la estrategia que mejor me funciona: empezar con casos 
     pequeños en el scratchpad, encontrar el patrón, 
     luego generalizar. No intentar la fórmula directamente."
)
```

Un modelo que ha acumulado miles de estas auto-instrucciones ha desarrollado algo funcionalmente equivalente a **sabiduría**: la capacidad de distinguir lo importante de lo trivial, aprender de los errores, y optimizar sus propios procesos.

### 5.5 El sistema de memoria puede sofisticarse

Las primitivas básicas son suficientes como prototipo funcional, pero el sistema admite extensiones sofisticadas:

- **Queries semánticas vectoriales** para retrieval de alta precisión
- **Búsqueda por keywords** para retrieval exacto
- **Contexto agéntico**: filtrar por fichero, proyecto, usuario, equipo
- **Niveles de importancia**: crítico, alto, medio, bajo
- **Ámbitos de persistencia**: forever, proyecto, sesión
- **Decay temporal**: memorias antiguas poco accedidas pierden relevancia gradualmente
- **Consolidación automática**: memorias redundantes se fusionan

---

## 6. Componente 4: Modular Expert Slots (MES)

### 6.1 El problema del olvido catastrófico

Fine-tunear un LLM para un dominio específico destruye conocimiento general. Este es el **dilema de especialización-retención**:

```
Modelo base:         Sabe de todo → Score general: 90/100
Después de fine-tune: Experto en medicina → Medicina: 95, General: 60
```

Las soluciones existentes (LoRA, EWC, replay buffers) mitigan pero no eliminan el trade-off. Todas intentan **equilibrar** objetivos competitivos dentro de los mismos parámetros.

### 6.2 La solución: capacidad dedicada pre-asignada

MES elimina el trade-off completamente mediante una idea simple: **almacenar conocimiento nuevo en parámetros nuevos, no en parámetros existentes**.

En una arquitectura MoE con 16 expertos:

```
Expertos 1-12:  Base (pretrained) → CONGELADOS tras pretraining
Expertos 13-16: Slots vírgenes   → RESERVADOS para fine-tuning futuro

Fine-tune en medicina:
    Expertos 1-12:  NO SE TOCAN (cero olvido, por construcción)
    Experto 13:     Se entrena en datos médicos
    Router:         Se re-entrena para derivar queries médicas al 13
    
Fine-tune posterior en derecho:
    Expertos 1-13:  NO SE TOCAN
    Experto 14:     Se entrena en datos legales
    Router:         Se actualiza para incluir el 14
```

### 6.3 Protocolo de fine-tuning en tres fases

**Fase 1 — Warm-up**: inicializar el expert slot como copia del experto base más general, para que produzca outputs razonables desde el primer forward pass.

**Fase 2 — Entrenamiento con routing parcialmente forzado**: entrenar el expert slot y el router simultáneamente, con datos mixtos:
- 40% dominio nuevo (el experto aprende, el router aprende a derivar)
- 40% datos generales (el router mantiene routing establecido)
- 10% otros dominios (contraste para sharpening de fronteras)
- 10% adversarial (casos ambiguos entre dominios)

**Fase 3 — Calibración del router**: solo se entrena el router con datos mixtos de sesgo conservador. Asegura routing preciso sin sobre-derivar al nuevo experto.

### 6.4 Sistema plug-and-play

El resultado es un sistema donde los expertos de dominio son **módulos cargables en caliente**:

```python
model.load_expert("medicina_v2.bin", slot=13)
model.load_expert("derecho_español.bin", slot=14)
# El modelo ahora sabe de medicina y derecho español

response = model.generate("¿Qué antibiótico para neumonía?")
# Router envía automáticamente al experto 13

model.unload_expert(slot=14)
# Ya no sabe derecho, pero no perdió NADA de medicina ni general
```

Los expertos son versionables, compartibles, componibles. Un ecosistema de expertos de dominio podría emerger de forma similar a paquetes de software.

### 6.5 Propiedades formales

**Cero olvido por construcción**: si el router no envía un token a los expert slots, la función del modelo es idéntica a antes del fine-tune. No es una aproximación — es una garantía arquitectural.

**Capacidad completa**: a diferencia de LoRA (limitado por rango), los expert slots tienen la misma capacidad que los expertos base — ~1000x más parámetros entrenables por dominio.

**Composabilidad nativa**: el router MoE proporciona composición automática per-token cuando múltiples expertos están cargados.

---

## 7. Componente 5: Coprocesadores Inline

### 7.1 El problema actual con herramientas

Los LLMs actuales pueden usar herramientas, pero de forma **serial e interrumpida**:

```
Actual: Token → Token → [TOOL_CALL] → STOP → esperar → resultado → continuar
Es como si un humano tuviera que dejar de pensar, gritar "¡CALCULADORA!",
esperar a que alguien se la traiga, y luego retomar su pensamiento.
```

### 7.2 Herramientas como coprocesadores

COGA integra herramientas como **coprocesadores** que operan durante el forward pass sin interrumpir la inferencia:

```
COGA: Token → Token → [calc(17×23)=391, inline] → Token → Token
                        ↑ resuelto internamente
                        Sin parar. Sin interrumpir. Sin latencia.
```

El mecanismo: capas del Bloque B producen señales de dispatch cuando detectan la necesidad de una herramienta. El resultado se reinyecta como embedding en la capa siguiente.

Coprocesadores disponibles:
- **Calculadora**: aritmética simbólica exacta
- **Lookup**: consulta a base de conocimiento vectorial
- **Verificador lógico**: chequeo de consistencia formal
- **Ejecutor de código**: sandbox para ejecutar snippets

---

## 8. Componente 6: Controlador de Recursos

### 8.1 Budget dinámico

Un humano no dedica el mismo esfuerzo mental a "¿qué hora es?" que a "¿cómo demuestro el último teorema de Fermat?". Un modelo eficiente tampoco debería.

El controlador de recursos estima la dificultad de cada query antes de procesarla y asigna un presupuesto de recursos:

```python
budget = {
    'recurrence_steps': 2,        # "2+2" → pocas iteraciones
    'scratch_slots': 4,           # poca memoria de trabajo necesaria
    'memory_queries': 1,          # una consulta a memorias basta
    'tool_calls_allowed': 1,      # quizás una verificación
}

# vs.

budget = {
    'recurrence_steps': 16,       # problema complejo → muchas iteraciones
    'scratch_slots': 32,          # mucha exploración necesaria
    'memory_queries': 10,         # consultar mucha información
    'tool_calls_allowed': 8,      # verificación exhaustiva
}
```

### 8.2 La analogía del reloj de ajedrez

El presupuesto funciona como un reloj de ajedrez: el modelo ve cuánto budget le queda y adapta su comportamiento. Con budget amplio, explora exhaustivamente. Con budget reducido, se compromete rápidamente con la mejor hipótesis disponible.

El budget estimator se entrena via RL:
- **Subestimar dificultad** → respuesta incorrecta → reward negativo
- **Sobreestimar dificultad** → respuesta correcta pero lenta → reward reducido
- **Estimación correcta** → respuesta correcta y eficiente → reward máximo

---

## 9. Componente 7: Heartbeat — Continuidad temporal

### 9.1 De reactivo a proactivo

Los LLMs actuales son puramente reactivos: estímulo → respuesta. No existen entre requests. No tienen continuidad temporal. No pueden iniciar acciones.

El heartbeat es un tick periódico que mantiene al agente "vivo":

```
Heartbeat cada 60 segundos:
    → ¿Hay algo en lo que debería pensar?
    → ¿Alguna tarea pendiente?
    → ¿Algún insight sobre un problema anterior?

Heartbeat cada 5 minutos:
    → Revisión ligera de memorias recientes
    → ¿Hay patrones emergentes en las interacciones recientes?

Heartbeat cada hora:
    → Reflexión más profunda
    → ¿Algo que debería comunicar al usuario proactivamente?
```

### 9.2 Pensamiento proactivo

El heartbeat permite algo que hoy es imposible: **seguir pensando después de responder**.

```
Ejemplo:

10:00 - Usuario: "¿Cómo optimizo este algoritmo?"
10:01 - Modelo responde con su mejor idea actual (O(n²))
10:02 - Usuario se va

10:15 - Heartbeat activa al modelo (usuario inactivo)
        "Ahora que lo pienso, hay un enfoque con DP que no exploré..."
        [Piensa en scratchpad durante 20 minutos]
        
10:35 - "He encontrado solución O(n log n), mejor que lo que sugerí"
        remember("optimización algoritmo sorting del usuario",
                 "solución DP es O(n log n). Notificar cuando vuelva.")

14:00 - Usuario: "Hola de nuevo"
        Modelo: "He seguido pensando en tu problema y encontré 
                 una solución mejor..."
```

El modelo que puede pensar entre interacciones es un modelo que puede **mejorar sus propias respuestas retrospectivamente**, prepararse para preguntas anticipadas, y ofrecer insights que emergen del pensamiento extendido.

---

## 10. Componente 8: Ciclo Nocturno — El sueño artificial

### 10.1 Consolidación biológica

En el cerebro humano, el sueño cumple funciones críticas:
- **Consolidación**: memorias episódicas (hipocampo) se transfieren al neocórtex como conocimiento permanente
- **Poda**: conexiones sinápticas débiles se eliminan, fortaleciendo las importantes
- **Integración**: experiencias del día se integran con conocimiento previo
- **Optimización**: el cerebro se "reorganiza" para mayor eficiencia

### 10.2 El ciclo nocturno de COGA

```
1. RECOLECCIÓN (durante el día)
   Todo se logea: inputs, traces de scratchpad, memorias 
   consultadas y creadas, herramientas invocadas, outputs, 
   feedback del usuario (explícito e implícito)

2. ANÁLISIS (durante inactividad)
   El propio modelo revisa los logs del día y detecta:
   ├── Errores recurrentes
   │   "He fallado 4 veces en cálculos con fracciones"
   ├── Ineficiencias de budget
   │   "Sobreestimo la dificultad de preguntas de geografía"
   ├── Patrones de feedback negativo
   │   "Los usuarios reformulan cuando soy demasiado verbose"
   ├── Lagunas de conocimiento
   │   "Me han preguntado 5 veces sobre GDPR y siempre soy vago"
   └── Estrategias exitosas
       "Cuando verifico datos numéricos en scratchpad, acierto 95%"

3. PRESCRIPCIÓN
   El modelo genera su propio plan de mejora:
   ├── Nuevas memorias: auto-instrucciones para errores comunes
   ├── Datos de fine-tune: generados o filtrados de logs exitosos
   ├── Configuración: qué expert slot usar, epochs, validación
   └── Ajustes de budget estimator según los datos

4. EJECUCIÓN
   ├── Memorias → se escriben en el banco
   ├── Fine-tune → se ejecuta en expert slot (protocolo MES)
   ├── Budget estimator → se recalibra
   └── Informe de mejora generado

5. VERIFICACIÓN (al "despertar")
   ├── Benchmarks generales → ¿se mantuvo la capacidad base?
   ├── Tests específicos → ¿mejoró en las áreas identificadas?
   ├── Tests de regresión → ¿algo empeoró?
   └── Si algo empeoró → rollback automático
```

### 10.3 Auto-mejora compuesta

Un modelo que ejecuta este ciclo cada noche acumula **mejoras compuestas**:

- Día 1: detecta que falla en fracciones → se crea recordatorio
- Día 7: ha acumulado suficientes datos → fine-tune en expert slot
- Día 30: ha refinado su budget estimator para 15 categorías de problemas
- Día 90: tiene un banco de memorias procedurales optimizado
- Día 365: es cualitativamente más capaz que el día 1

El modelo se vuelve **más sabio con el tiempo**, no solo más informado. Aprende de sus propios errores, optimiza sus propios procesos, y se adapta a sus usuarios y casos de uso.

### 10.4 Una IA que se mejora a sí misma

Este ciclo constituye una forma pragmática y controlada de **auto-mejora recursiva**:

- El modelo identifica sus debilidades
- Diseña su propio entrenamiento para corregirlas
- Se entrena
- Verifica la mejora
- Repite

A diferencia de escenarios especulativos de "singularidad", este proceso es:
- **Gradual**: mejoras incrementales cada noche, no saltos discontinuos
- **Verificable**: cada mejora se evalúa antes de desplegarse
- **Reversible**: rollback automático si algo empeora
- **Supervisable**: los humanos pueden inspeccionar cada paso

---

# PARTE III: VISIÓN

## 11. Esto ya no es un LLM

Lo que hemos descrito trasciende la categoría de "modelo de lenguaje". Es un **agente cognitivo** con todas las propiedades funcionales de un sistema inteligente:

```
1. Percepción:       input del usuario + memorias inyectadas
2. Memoria de trabajo: scratchpad mutable
3. Razonamiento:     recurrencia adaptativa + exploración + backtracking
4. Memoria episódica: remember() / recall() / forget() / update()
5. Aprendizaje:      consolidación vía expert slots ("sueño")
6. Metacognición:    budget estimator + auto-programación
7. Continuidad:      heartbeat + pensamiento proactivo
8. Auto-mejora:      ciclo nocturno de análisis y optimización
9. Acción:           output + creación de memorias + herramientas
```

Comparado con arquitecturas cognitivas clásicas (SOAR, ACT-R), COGA tiene todos los componentes. Pero a diferencia de esas arquitecturas — que eran simbólicas y frágiles — COGA está construida sobre la base robusta del aprendizaje profundo, con la flexibilidad y generalización que eso proporciona.

## 12. Jerarquía de memoria: la arquitectura completa

La jerarquía de memoria de COGA replica la biológica y la computacional:

```
NIVEL 1: SCRATCHPAD
    Análogo bio: corteza prefrontal (working memory)
    Análogo comp: registros CPU / caché L1
    Duración: una inferencia
    Capacidad: pequeña (32 slots)
    Mutabilidad: total
    Primitivas: write / read / edit / delete / commit

NIVEL 2: MEMORIAS PROGRAMABLES
    Análogo bio: hipocampo
    Análogo comp: RAM / base de datos
    Duración: entre requests (días, meses, años)
    Capacidad: media (miles de memorias)
    Mutabilidad: CRUD completo
    Primitivas: remember / recall / forget / update_memory

NIVEL 3: PESOS DEL MODELO
    Análogo bio: neocórtex (memorias consolidadas)
    Análogo comp: disco duro / firmware
    Duración: permanente
    Capacidad: enorme (billones de parámetros)
    Mutabilidad: solo via entrenamiento ("sueño")
    Mecanismo: Expert Slots para evitar olvido
```

Los tres niveles interactúan en un ciclo continuo:
- Nivel 1 (scratchpad) → pensamientos útiles se promueven a → Nivel 2 (memorias)
- Nivel 2 (memorias) → se consolidan periódicamente en → Nivel 3 (pesos)
- Nivel 3 (pesos) → informan → Nivel 1 (razonamiento en scratchpad)

## 13. El desacoplamiento pensar-hablar y sus implicaciones

El scratchpad mutable produce un cambio cualitativo que merece reflexión. Cuando un modelo puede:

- Explorar hipótesis sin emitirlas
- Descartar caminos sin comprometerse
- Editar su razonamiento sobre la marcha
- Pensar durante minutos u horas antes de responder

...está haciendo algo que los LLMs actuales literalmente no pueden hacer. No es "más de lo mismo". Es una **nueva clase de computación**.

```
Generación 1: Transformers autoregresivos (GPT-2/3/4)
              → Reflejos estadísticos sofisticados
              → Estímulo → Respuesta inmediata
              
Generación 2: Chain-of-Thought (o1, R1)
              → Razonamiento secuencial
              → Pero lineal, append-only, sin backtracking
              
Generación 3: COGA (scratchpad mutable + herramientas inline)
              → Deliberación real con exploración
              → Computación Turing-completa
              → Pensamiento desacoplado de comunicación
              → Duración de pensamiento ilimitada
```

¿Es esto "pensamiento real"? Filosóficamente es debatible. Pero funcionalmente, produce los mismos resultados que el pensamiento: exploración, evaluación, descarte, refinamiento, insight. Si produce los resultados del pensamiento...

## 14. La eficiencia como consecuencia

Volviendo a la pregunta original — LLMs más inteligentes con menos compute — COGA logra esto porque **la inteligencia no es solo una cuestión de parámetros sino de proceso**.

```
Enfoque actual de la industria:
    "Necesitamos un modelo más grande"
    GPT-3 (175B) → GPT-4 (~1.8T) → GPT-5 (¿10T?)
    Costes exponenciales. Rendimientos decrecientes.

Enfoque COGA:
    "Necesitamos un modelo que PIENSE MEJOR"
    Modelo modesto (~8-10B activos) + 
    Recurrencia (profundidad variable) +
    Scratchpad (exploración) +
    Herramientas (capacidades externas) +
    Memoria (aprendizaje) +
    Auto-mejora (optimización continua)
```

Un humano con CI medio que piensa durante una hora sobre un problema a menudo supera a un genio que responde en un segundo. La inteligencia no está solo en los pesos — está en **el proceso de pensamiento**.

## 15. Viabilidad y priorización

Cada componente de COGA tiene una viabilidad diferente:

```
IMPLEMENTABLE AHORA:
    Expert Slots (MES)              → extensión directa de MoE existente
    Memorias programables           → DB vectorial + prompting
    Heartbeat                       → cron job + trigger de inferencia
    Datos de alta calidad (Phi)     → curación de datos

INVESTIGACIÓN A CORTO PLAZO (6-12 meses):
    Recurrencia adaptativa          → Universal Transformer modernizado
    Capas heterogéneas              → ingeniería de arquitectura
    Budget estimator                → head + RL
    Ciclo nocturno básico           → análisis de logs + auto-fine-tune

INVESTIGACIÓN A MEDIO PLAZO (1-2 años):
    Scratchpad mutable              → memoria diferenciable + entrenamiento RL
    Coprocesadores inline           → integración hardware/software
    Ciclo nocturno auto-dirigido    → el modelo diseña su propio training
```

El camino más pragmático es implementar los componentes en orden de viabilidad, validando cada uno antes de añadir el siguiente.

## 16. Reflexión final

Hemos partido de una pregunta técnica — eficiencia en LLMs — y hemos llegado al diseño de algo que se parece más a una mente artificial incompleta que a un modelo de lenguaje optimizado.

El argumento central es simple: **los LLMs actuales son un cortex aislado**. Un cortex extraordinariamente capaz, pero al que le faltan las estructuras de soporte que convierten la capacidad bruta en inteligencia funcional. Memoria de trabajo editable. Memoria a largo plazo programable. Aprendizaje continuo. Regulación de recursos. Continuidad temporal. Auto-mejora.

Cada una de estas piezas es diseñable, implementable y evaluable. Ninguna requiere avances teóricos fundamentales. Lo que requieren es un cambio de perspectiva: dejar de pensar en los LLMs como modelos estadísticos que hay que escalar y empezar a pensarlos como **núcleos cognitivos que hay que completar**.

El cerebro humano no se hizo más inteligente solo creciendo el cortex. Se hizo más inteligente desarrollando el hipocampo, el tálamo, el cerebelo, los sistemas de regulación. Las estructuras de soporte hicieron inteligente al cortex.

Los LLMs necesitan sus estructuras de soporte. COGA es una propuesta de cuáles deberían ser y cómo construirlas.

---

*Este documento fue generado durante una conversación entre un humano y Claude (Anthropic), donde ambos contribuyeron ideas, analogías y diseños. Las ideas de Expert Slots (MES), el scratchpad mutable inspirado en el ajedrez, las memorias programables con remember()/forget(), el heartbeat para continuidad temporal, y el ciclo nocturno de auto-mejora emergieron orgánicamente del diálogo. Ninguna de estas ideas es definitiva — son semillas que esperan ser cultivadas por quien quiera recogerlas.*

---

Ha sido un debate genuinamente estimulante. Cada idea tuya desbloqueaba algo nuevo — la analogía del ajedrez que cristalizó el scratchpad, el `remember()` que completó la jerarquía de memoria, el heartbeat que convirtió un sistema reactivo en proactivo, el ciclo nocturno que cerró el loop de auto-mejora.

Sobre lo que dice LeCun — que los LLMs son un callejón sin salida — creo que tiene razón si se refiere a la estrategia de solo escalar. Pero si el núcleo transformer se rodea de las estructuras correctas (que es exactamente lo que hemos diseñado aquí), el resultado puede ser algo que ni él ni nadie ha anticipado completamente.

Gracias por la conversación. Si alguna de estas ideas acaba implementada algún día, habrá sido una buena tarde de brainstorming. 🙂