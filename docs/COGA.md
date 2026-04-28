

# Research Proposal: Cognitive OS Architecture for Efficient Language Models

---

## COGA: Cognitive Operating System Architecture for Language Models

### Towards More Intelligent LLMs with Less Parameters and Compute

---

## 1. Resumen Ejecutivo

### 1.1 Problema

Los Large Language Models (LLMs) actuales siguen una estrategia de escalado fundamentalmente ineficiente: más parámetros, más datos, más compute. GPT-4 se estima en ~1.8 trillones de parámetros. Llama 3.1 tiene 405B. Cada generación multiplica los costes de entrenamiento e inferencia. Sin embargo, el cerebro humano — con ~86 mil millones de neuronas, ~20W de consumo, y activación simultánea de solo el 1-5% de sus neuronas — supera a estos modelos en razonamiento general, eficiencia energética y capacidad de aprendizaje continuo.

Esta propuesta argumenta que el cuello de botella no es el tamaño, sino la **arquitectura cognitiva**. Los LLMs actuales carecen de:

- **Memoria de trabajo editable** (piensan "en voz alta", sin posibilidad de explorar y descartar)
- **Computación adaptativa** (gastan el mismo compute en "2+2" que en "demuestra el teorema de Fermat")
- **Modularidad real** (no pueden adquirir conocimiento nuevo sin riesgo de olvido catastrófico)
- **Herramientas integradas** (delegan a sistemas externos con interrupciones costosas)
- **Recurrencia** (están limitados a un número fijo de pasos de razonamiento)

### 1.2 Propuesta

Presentamos **COGA** (Cognitive Operating System Architecture), una arquitectura integrada que trata al LLM como un sistema operativo cognitivo con componentes especializados y coordinados:

| Componente | Análogo | Función |
|---|---|---|
| Núcleo recurrente heterogéneo | CPU | Procesamiento central con profundidad adaptativa |
| Scratchpad mutable | RAM | Memoria de trabajo editable, pensamiento sin emisión |
| Memoria episódica/semántica | Disco + Caché | Almacenamiento y recuperación asociativa mid-inference |
| Coprocesadores inline | FPU/GPU | Herramientas integradas sin interrumpir inferencia |
| MoE con slots modulares | Microservicios | Conocimiento especializado plug-and-play |
| Controlador de recursos | Scheduler del OS | Budget dinámico de compute por query |

### 1.3 Contribuciones principales

1. **Scratchpad Mutable**: un mecanismo de memoria de trabajo interna que permite al modelo explorar hipótesis, editarlas y descartarlas antes de emitir una respuesta, con un protocolo de entrenamiento en 4 fases (datos sintéticos → SFT → RL → self-play).

2. **Modular Expert Slots (MES)**: un protocolo para reservar expertos vacíos en arquitecturas MoE que permite fine-tuning sin olvido catastrófico, con sistema plug-and-play de conocimiento.

3. **Recurrencia Adaptativa con Budget Aprendido**: un mecanismo que permite al modelo reusar bloques de capas un número variable de veces, con presupuesto de compute estimado por el propio modelo y entrenado via RL.

4. **Inline Tool Fusion**: integración de herramientas como coprocesadores que operan durante el forward pass sin interrumpir la inferencia.

5. **Arquitectura Heterogénea por Capas**: abandono de la homogeneidad de capas en favor de una estructura funcional con diferentes tamaños y tipos de capa según su posición y rol.

### 1.4 Resultado esperado

Un modelo de **~7-15B parámetros activos** (sobre ~50-80B totales en arquitectura MoE) que compita con modelos de 70-400B parámetros densos en benchmarks de razonamiento, con:

- **3-5x menos compute en inferencia** para queries simples
- **Capacidad de razonamiento profundo** comparable a modelos con chain-of-thought externo
- **Fine-tuning modular** sin olvido catastrófico
- **Aprendizaje continuo** via slots de expertos

---

## 2. Arquitectura Propuesta Integrada

### 2.1 Visión general

```
                    ┌─────────────────────────────────┐
                    │     CONTROLADOR DE RECURSOS     │
                    │   (Budget Estimator + Scheduler)│
                    └──────────────┬──────────────────┘
                                   │ budget asignado
                    ┌──────────────▼──────────────────┐
 Input ───────────► │      ENCODER / EMBEDDING        │
                    │   (Tokenización multi-escala)   │
                    └──────────────┬──────────────────┘
                                   │
              ┌────────────────────▼──────────────────────┐
              │         NÚCLEO RECURRENTE HETEROGÉNEO     │
              │                                           │
              │  ┌─────────────────────────────────────┐  │
              │  │  BLOQUE A: Parsing (SSM, pequeño)   │  │
              │  │  Capas 1-4: d=2048, Mamba-style     │  │
              │  └──────────────┬──────────────────────┘  │
              │                 │                         │
              │  ┌──────────────▼──────────────────────┐  │
              │  │  BLOQUE B: Razonamiento (grande)    │  │
              │  │  Capas 5-16: d=6144, Transformer    │  │
              │  │  + MoE con Expert Slots modulares   │  │
              │  │  + Cross-attention a Scratchpad     │  │
              │  │  + Cross-attention a Memoria        │  │
              │  │  + Dispatch a Coprocesadores        │  │
              │  └──────────────┬──────────────────────┘  │
              │                 │                         │
              │        ┌────────▼─────────┐               │
              │        │ ¿Otra iteración? │◄── Halt Head  │
              │        │  (recurrencia)   │               │
              │        └───┬─────────┬────┘               │
              │         SÍ │         │ NO                 │
              │            │         │                    │
              │  ┌─────────▼──┐      │                    │
              │  │ Loop back  │      │                    │
              │  │ al Bloque B│      │                    │
              │  └────────────┘      │                    │
              │                      │                    │
              │  ┌───────────────────▼─────────────────┐  │
              │  │  BLOQUE C: Refinamiento + Output    │  │
              │  │  Capas 17-22: d=4096, Transformer   │  │
              │  └──────────────┬──────────────────────┘  │
              └─────────────────┼─────────────────────────┘
                                │
                    ┌───────────▼──────────┐
                    │    OUTPUT / DECODE   │
                    └──────────────────────┘

    ══════════ COMPONENTES LATERALES ══════════

  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐
  │ SCRATCHPAD   │  │   MEMORIA    │  │ COPROCESADORES  │
  │ MUTABLE      │  │  EPISÓDICA/  │  │                 │
  │              │  │  SEMÁNTICA   │  │ • Calculadora   │
  │ Banco KV     │  │              │  │ • DB lookup     │
  │ externo con  │  │ Vector store │  │ • Verificador   │
  │ read/write/  │  │ con pattern  │  │   lógico        │
  │ edit/delete  │  │ completion   │  │ • Code executor │
  │              │  │              │  │                 │
  └──────┬───────┘  └──────┬───────┘  └────────┬────────┘
         │                 │                   │
         └─────────────────┼───────────────────┘
                           │
              Cross-attention bidireccional
              con el Bloque B (capas 5-16)
```

### 2.2 Componente 1: Núcleo Recurrente Heterogéneo

#### 2.2.1 Estructura de bloques

El núcleo abandona la arquitectura homogénea (todas las capas iguales) en favor de tres bloques funcionales diferenciados:

**Bloque A — Parsing y Representación Inicial**
```
- Tipo: State Space Model (Mamba-2 style)
- Dimensión: d_model = 2048 (pequeño)
- Capas: 4
- Función: procesamiento secuencial eficiente, 
  construcción de representaciones básicas
- Justificación: las primeras capas de un transformer 
  procesan features sintácticos/superficiales que no 
  requieren atención cuadrática completa
- Sin recurrencia: se ejecuta una sola vez
```

**Bloque B — Razonamiento Profundo (Recurrente)**
```
- Tipo: Transformer con atención densa + MoE
- Dimensión: d_model = 6144 (grande)
- Capas: 12 
- Heads de atención: 48
- MoE: 16 expertos por capa, top-2 routing
  - 12 expertos base (preentrenados, congelables)
  - 4 expert slots modulares (reservados)
- Cross-attention: al scratchpad, memoria, coprocesadores
- RECURRENTE: este bloque puede ejecutarse N veces
- Halt head: cada iteración produce p(halt) ∈ [0,1]
```

**Bloque C — Refinamiento y Formateo de Salida**
```
- Tipo: Transformer estándar
- Dimensión: d_model = 4096 (medio)
- Capas: 6
- Función: convertir representaciones internas en 
  distribución sobre vocabulario
- Sin recurrencia: se ejecuta una sola vez
```

#### 2.2.2 Mecanismo de recurrencia del Bloque B

```python
def forward_block_b(x, budget, scratchpad, memory, tools):
    """
    x: activaciones del Bloque A
    budget: número estimado de iteraciones (learned)
    """
    hidden = x
    cumulative_halt = 0.0
    all_outputs = []
    weights = []
    
    for step in range(budget):  # hard cap = budget estimado
        # Forward por las 12 capas del Bloque B
        for layer in block_b_layers:
            hidden = layer.self_attention(hidden)
            hidden = layer.cross_attention_scratchpad(hidden, scratchpad)
            hidden = layer.cross_attention_memory(hidden, memory)
            hidden = layer.moe_ffn(hidden)  # routing a expertos
            hidden = layer.tool_dispatch(hidden, tools)  # coprocesadores
        
        # ¿Debería parar?
        halt_prob = halt_head(hidden)
        cumulative_halt += halt_prob
        
        all_outputs.append(hidden)
        weights.append(halt_prob)
        
        # Convergencia detectada
        if step > 0:
            delta = cosine_similarity(all_outputs[-1], all_outputs[-2])
            if delta > 0.99:  # no está cambiando
                break
        
        # Halting aprendido
        if cumulative_halt > 1.0 - epsilon:
            break
    
    # Output ponderado
    final = sum(w * o for w, o in zip(normalize(weights), all_outputs))
    return final, step + 1  # devuelve también los pasos usados
```

#### 2.2.3 Parámetros estimados

```
Bloque A (SSM):          ~0.5B parámetros
Bloque B (Transformer+MoE): 
  - Atención:            ~3B parámetros
  - MoE (16 expertos):   ~40B parámetros totales
  - Activos (top-2):     ~5B parámetros por forward
Bloque C (Transformer):  ~2B parámetros

TOTAL:                   ~45B parámetros totales
ACTIVOS por forward:     ~8-10B parámetros
Con recurrencia x4:      ~8-10B params × 4 = compute de ~35B denso
                         Pero con la profundidad de razonamiento 
                         de un modelo mucho mayor
```

### 2.3 Componente 2: Scratchpad Mutable

#### 2.3.1 Arquitectura del Scratchpad

```
┌──────────────────────────────────────┐
│         SCRATCHPAD BANK              │
│                                      │
│  Implementación: External Key-Value  │
│  Memory con addressing por contenido │
│                                      │
│  Slots: S (configurable, e.g. 32)    │
│  Dimensión por slot: d_model = 6144  │
│                                      │
│  Cada slot contiene:                 │
│    key:     vector de d_k            │
│    value:   vector de d_v            │
│    status:  {empty, written, locked} │
│    version: int (para tracking)      │
│    confidence: float [0,1]           │
│                                      │
│  Operaciones disponibles:            │
│    WRITE(content) → nuevo slot       │
│    READ(query)    → content matching │
│    EDIT(slot_id, new_content)        │
│    DELETE(slot_id)                   │
│    COMPARE(slot_a, slot_b) → score   │
│    COMMIT(slot_id) → marcar final    │
│    CLEAR_ALL() → reset completo      │
└──────────────────────────────────────┘
```

#### 2.3.2 Interacción con el Bloque B

En cada capa del Bloque B, después de self-attention:

```python
class BlockBLayer(nn.Module):
    def forward(self, hidden, scratchpad):
        # 1. Self-attention normal
        hidden = self.self_attention(hidden)
        
        # 2. Scratchpad interaction head
        # Produce un vector de "intención" sobre el scratchpad
        scratch_intent = self.scratch_head(hidden)
        # scratch_intent codifica: operación + contenido
        
        operation = classify_operation(scratch_intent)
        
        if operation == WRITE:
            content = self.scratch_write_proj(scratch_intent)
            scratchpad.write(content)
        
        elif operation == READ:
            query = self.scratch_read_proj(scratch_intent)
            retrieved = scratchpad.read(query)  # content-based addressing
            # Inyectar lo recuperado via cross-attention
            hidden = hidden + self.cross_attn(hidden, retrieved)
        
        elif operation == EDIT:
            slot_id = scratchpad.find_closest(scratch_intent)
            new_content = self.scratch_edit_proj(scratch_intent)
            scratchpad.edit(slot_id, new_content)
        
        elif operation == NOP:
            pass  # no hacer nada con el scratchpad
        
        # 3. Resto de la capa (MoE FFN, etc.)
        hidden = self.moe_ffn(hidden)
        
        return hidden
```

#### 2.3.3 Protocolo de entrenamiento del Scratchpad (4 fases)

```
FASE 1: Dataset sintético
├── Modelo frontera (e.g. Claude, GPT-4) genera traces
│   con uso explícito de scratchpad en formato estructurado
├── Dominios: matemáticas, lógica, coding, análisis, planificación
├── ~500K-1M ejemplos con traces de scratchpad
├── Filtrado por calidad: solo ejemplos donde el scratchpad
│   fue genuinamente útil (respuesta correcta que requirió
│   exploración/corrección)
└── Formato estandarizado de operaciones

FASE 2: Supervised Fine-Tuning (SFT)
├── Entrenar el modelo COGA con estos traces
├── Loss: predicción de operaciones de scratchpad + respuesta final
├── El modelo aprende la MECÁNICA del scratchpad
├── No necesita ser óptimo, solo funcional
└── ~2-5 epochs sobre el dataset sintético

FASE 3: Reinforcement Learning
├── Reward function:
│   R = R_accuracy + R_efficiency + R_self_correction
│   
│   R_accuracy = 1.0 si respuesta correcta, 0.0 si no
│   R_efficiency = -0.01 × num_operaciones_scratchpad
│   R_self_correction = +0.1 si editó/borró hipótesis incorrecta
│   R_unused_write = -0.05 si escribió pero nunca leyó
│   
├── Algoritmo: PPO o GRPO sobre trajectories completas
├── El modelo aprende CUÁNDO usar el scratchpad
│   (no siempre, solo cuando ayuda)
└── ~100K-500K episodes de RL

FASE 4: Self-play iterativo
├── El modelo resuelve problemas nuevos con scratchpad
├── Filtra por: respuesta correcta + uso eficiente del scratchpad
├── Re-entrena con los mejores ejemplos auto-generados
├── Repite 3-5 iteraciones
└── Resultado: uso del scratchpad progresivamente más sofisticado
```

### 2.4 Componente 3: Memoria Episódica/Semántica

#### 2.4.1 Arquitectura

```
┌─────────────────────────────────────────────┐
│           MEMORY SYSTEM                     │
│                                             │
│  ┌─────────────────────────────────┐        │
│  │    MEMORIA SEMÁNTICA            │        │
│  │    (hechos, conocimiento)       │        │
│  │                                 │        │
│  │    Vector store persistente     │        │
│  │    Actualizable sin retraining  │        │
│  │    Retrieval por similitud      │        │
│  └─────────────────────────────────┘        │
│                                             │
│  ┌─────────────────────────────────┐        │
│  │    MEMORIA EPISÓDICA            │        │
│  │    (experiencias, contexto)     │        │
│  │                                 │        │
│  │    Conversaciones anteriores    │        │
│  │    Patrones de error pasados    │        │
│  │    Preferencias del usuario     │        │
│  └─────────────────────────────────┘        │
│                                             │
│  ┌─────────────────────────────────┐        │
│  │    MEMORIA PROCEDURAL           │        │
│  │    (cómo hacer cosas)           │        │
│  │                                 │        │
│  │    Templates de razonamiento    │        │
│  │    Estrategias exitosas previas │        │
│  │    Patrones de uso de tools     │        │
│  └─────────────────────────────────┘        │
│                                             │
│  Trigger: pattern completion automático     │
│  Las activaciones del Bloque B disparan     │
│  búsqueda asociativa en las 3 memorias      │
│  Resultados inyectados via cross-attention  │
└─────────────────────────────────────────────┘
```

#### 2.4.2 Mecanismo de retrieval mid-inference

```python
class MemoryRetrievalLayer(nn.Module):
    """Se inserta en capas específicas del Bloque B (e.g. capas 8, 12)"""
    
    def forward(self, hidden, memory_bank):
        # 1. Generar query de memoria desde activaciones actuales
        memory_query = self.memory_query_proj(hidden)  # d_model → d_memory
        
        # 2. Estimar relevancia: ¿necesito consultar memoria?
        gate = sigmoid(self.memory_gate(hidden))  # [0, 1]
        
        if gate > threshold:  # Solo consultar si parece relevante
            # 3. Buscar en las tres memorias
            sem_results = memory_bank.semantic.search(memory_query, top_k=3)
            epi_results = memory_bank.episodic.search(memory_query, top_k=2)
            proc_results = memory_bank.procedural.search(memory_query, top_k=1)
            
            # 4. Combinar resultados
            all_memories = concat(sem_results, epi_results, proc_results)
            
            # 5. Cross-attention: hidden atiende a memorias recuperadas
            memory_context = self.cross_attention(
                query=hidden,
                key=all_memories.keys,
                value=all_memories.values
            )
            
            # 6. Gated injection
            hidden = hidden + gate * memory_context
        
        return hidden
```

### 2.5 Componente 4: Coprocesadores Inline

#### 2.5.1 Diseño

```
┌──────────────────────────────────────────────┐
│            COPROCESSOR BANK                  │
│                                              │
│  ┌────────────┐  ┌────────────┐  ┌─────────┐ │
│  │ CALC       │  │ LOOKUP     │  │ VERIFY  │ │
│  │            │  │            │  │         │ │
│  │ Aritmética │  │ DB/Wiki    │  │ Lógica  │ │
│  │ simbólica  │  │ lookup     │  │ formal  │ │
│  │            │  │            │  │         │ │
│  │ Input:     │  │ Input:     │  │ Input:  │ │
│  │ expresión  │  │ query      │  │ premisas│ │
│  │ math       │  │ semántica  │  │ + claim │ │
│  │            │  │            │  │         │ │
│  │ Output:    │  │ Output:    │  │ Output: │ │
│  │ resultado  │  │ fragmento  │  │ T/F/    │ │
│  │ numérico   │  │ relevante  │  │ unknown │ │
│  └─────┬──────┘  └─────┬──────┘  └────┬────┘ │
│        └───────────────┼──────────────┘      │
│                        │                     │
│  Interfaz: embedding → dispatch → embedding  │
│  Los resultados se reinyectan como vectores  │
│  en la capa siguiente del Bloque B           │
└──────────────────────────────────────────────┘
```

#### 2.5.2 Mecanismo de dispatch

```python
class ToolDispatchLayer(nn.Module):
    """Se inserta después del MoE FFN en capas del Bloque B"""
    
    def __init__(self, tools):
        self.tool_classifier = nn.Linear(d_model, len(tools) + 1)  # +1 = NOP
        self.tool_input_projs = nn.ModuleDict({
            name: nn.Linear(d_model, tool.input_dim) 
            for name, tool in tools.items()
        })
        self.tool_output_projs = nn.ModuleDict({
            name: nn.Linear(tool.output_dim, d_model)
            for name, tool in tools.items()
        })
    
    def forward(self, hidden, tools):
        # 1. ¿Necesito una herramienta? ¿Cuál?
        tool_logits = self.tool_classifier(hidden)
        tool_choice = gumbel_softmax(tool_logits)  # differentiable
        
        if tool_choice != NOP:
            # 2. Preparar input para la herramienta
            tool_input = self.tool_input_projs[tool_choice](hidden)
            
            # 3. Ejecutar herramienta (determinístico, no diferenciable)
            tool_output = tools[tool_choice].execute(tool_input)
            
            # 4. Proyectar resultado de vuelta al espacio del modelo
            tool_embedding = self.tool_output_projs[tool_choice](tool_output)
            
            # 5. Gated injection
            gate = sigmoid(self.tool_gate(hidden))
            hidden = hidden + gate * tool_embedding
        
        return hidden
```

### 2.6 Componente 5: Modular Expert Slots (MES)

#### 2.6.1 Estructura MoE con slots reservados

```
Cada capa MoE en el Bloque B:

┌─────────────────────────────────────────────┐
│                  ROUTER                     │
│    Input: token embedding (d=6144)          │
│    Output: top-2 expert selection           │
│    Parámetros: entrenables siempre          │
└──────────────────┬──────────────────────────┘
                   │ routing weights
    ┌──────────────┼──────────────────────┐
    ▼              ▼              ▼       ▼
┌────────┐  ┌────────┐    ┌────────┐ ┌────────┐
│Expert 1│  │Expert 2│... │Expert12│ │Expert13│
│ BASE   │  │ BASE   │    │ BASE   │ │ SLOT A │
│ 🔒     │  │ 🔒     │    │ 🔒    │ │ 🟢     │
└────────┘  └────────┘    └────────┘ └────────┘
                                      ┌────────┐
                                      │Expert14│
                                      │ SLOT B │
                                      │ 🟢     │
                                      └────────┘
                                      ┌────────┐
                                      │Expert15│
                                      │ SLOT C │
                                      │ 🟢     │
                                      └────────┘
                                      ┌────────┐
                                      │Expert16│
                                      │ SLOT D │
                                      │ 🟢     │
                                      └────────┘

🔒 = Congelado después del pretraining
🟢 = Disponible para fine-tuning modular
```

#### 2.6.2 Protocolo de Fine-Tuning Modular

```
PASO 1: INICIALIZACIÓN
   Expert slot 13 ← copy(Expert 7)  // Copiar experto más general
   Router weights: sin cambios

PASO 2: ENTRENAMIENTO
   Dataset: 40% dominio + 40% general + 10% otros + 10% adversarial
   Parámetros actualizados: Expert 13 + Router
   Parámetros congelados: Experts 1-12 + Bloques A,C + Scratchpad heads
   Router constraint: 70% dominio → Expert 13, 30% libre
   Epochs: 3-5

PASO 3: CALIBRACIÓN
   Dataset: mixto general + dominio
   Solo se entrena: Router
   Objetivo: routing suave y preciso sin collapse
   Epochs: 1-2

PASO 4: VALIDACIÓN
   Verificar:
   ✓ Accuracy en dominio nuevo ≥ target
   ✓ Accuracy en benchmarks generales sin degradación (< 1% drop)
   ✓ Router envía queries de dominio al expert 13 con alta probabilidad
   ✓ Router NO envía queries generales al expert 13
```

#### 2.6.3 Sistema Plug-and-Play

```python
class ExpertSlotManager:
    """Gestiona la carga/descarga de expertos modulares"""
    
    def __init__(self, base_model, num_base=12, num_slots=4):
        self.base_experts = base_model.experts[:num_base]  # congelados
        self.slots = {i: None for i in range(num_base, num_base + num_slots)}
        self.router = base_model.router
        self.slot_metadata = {}
    
    def load_expert(self, slot_id, expert_path, domain_name, 
                    router_patch_path=None):
        """Carga un experto entrenado en un slot"""
        self.slots[slot_id] = load_weights(expert_path)
        self.slot_metadata[slot_id] = {
            'domain': domain_name,
            'version': extract_version(expert_path),
            'loaded_at': timestamp()
        }
        if router_patch_path:
            self.router.apply_patch(slot_id, load_weights(router_patch_path))
    
    def unload_expert(self, slot_id):
        """Descarga un experto sin afectar nada más"""
        self.slots[slot_id] = None
        del self.slot_metadata[slot_id]
        self.router.remove_patch(slot_id)
    
    def list_loaded(self):
        """Lista los expertos actualmente cargados"""
        return {sid: meta for sid, meta in self.slot_metadata.items()}
    
    def swap_expert(self, slot_id, new_expert_path, new_domain):
        """Intercambia un experto por otro en caliente"""
        self.unload_expert(slot_id)
        self.load_expert(slot_id, new_expert_path, new_domain)
```

### 2.7 Componente 6: Controlador de Recursos (Budget Estimator)

#### 2.7.1 Arquitectura

```python
class ResourceController(nn.Module):
    """
    Estima el budget de compute necesario ANTES de procesar.
    Opera sobre los embeddings iniciales del input.
    """
    
    def __init__(self, d_model, max_recurrence, max_scratch_slots):
        # Lightweight network para estimar dificultad
        self.difficulty_estimator = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 0.0 = trivial, 1.0 = muy difícil
        )
        
        # Mapeo de dificultad a recursos
        self.max_recurrence = max_recurrence
        self.max_scratch_slots = max_scratch_slots
    
    def forward(self, input_embeddings):
        # Pool sobre la secuencia
        pooled = input_embeddings.mean(dim=1)  # [batch, d_model]
        
        # Estimar dificultad
        difficulty = self.difficulty_estimator(pooled)  # [0, 1]
        
        # Asignar recursos
        budget = {
            'recurrence_steps': max(1, int(difficulty * self.max_recurrence)),
            'scratch_slots': max(4, int(difficulty * self.max_scratch_slots)),
            'memory_queries': max(1, int(difficulty * 10)),
            'tool_calls_allowed': max(0, int(difficulty * 5)),
            'estimated_difficulty': difficulty.item()
        }
        
        return budget
```

#### 2.7.2 Entrenamiento del Budget Estimator

```python
# Se entrena junto con el modelo principal via RL

def compute_budget_reward(difficulty_estimate, steps_used, 
                          max_steps, answer_correct):
    """
    Reward que incentiva estimación precisa del budget
    """
    if answer_correct:
        # Correcta: premiar si el budget fue ajustado
        efficiency = 1.0 - (steps_used / max_steps)
        accuracy_bonus = 1.0
        
        # Penalizar sobreestimación (budget >> steps_used)
        waste = max(0, (difficulty_estimate * max_steps - steps_used) / max_steps)
        waste_penalty = -0.1 * waste
        
        return accuracy_bonus + 0.3 * efficiency + waste_penalty
    
    else:
        # Incorrecta: ¿fue por falta de budget?
        if steps_used == int(difficulty_estimate * max_steps):
            # Usó todo el budget y aún así falló → subestimó dificultad
            return -1.0  # penalización fuerte
        else:
            # No usó todo el budget → el problema no es el budget
            return -0.5
```

---

## 3. Plan de Implementación por Fases

### Fase 0 — Fundamentos (Meses 1-2)

```
Objetivo: infraestructura base y validación de componentes individuales

Tareas:
├── Implementar framework de entrenamiento modular
│   (cada componente puede entrenarse/evaluarse independientemente)
├── Crear benchmark suite personalizado que mida:
│   - Razonamiento multi-paso
│   - Eficiencia de compute
│   - Retención de conocimiento tras fine-tune
│   - Uso de herramientas
├── Establecer baselines:
│   - Transformer denso equivalente (~8B params)
│   - MoE estándar equivalente (~45B total, ~8B activos)
│   - Modelo con CoT externo
└── Infraestructura de logging detallado:
    - Visualización del scratchpad en tiempo real
    - Métricas de routing por experto
    - Tracking de pasos de recurrencia por query

Entregable: framework + baselines + benchmarks
```

### Fase 1 — MoE con Expert Slots (Meses 2-4)

```
Objetivo: validar Modular Expert Slots de forma aislada

Pasos:
├── Entrenar modelo MoE base (16 expertos, 45B total)
│   con datos generales de alta calidad (Phi-style)
├── Reservar 4 expert slots (13-16)
├── Implementar protocolo de fine-tune modular
│   - Test domain 1: Medicina (PubMedQA, MedQA)
│   - Test domain 2: Derecho (LegalBench)
│   - Test domain 3: Código (HumanEval, MBPP)
├── Evaluar:
│   - Accuracy en dominio vs. fine-tune completo
│   - Retención en benchmarks generales (MMLU, HellaSwag)
│   - Calidad del routing (¿envía al experto correcto?)
│   - Composabilidad (¿medicina + derecho funcionan juntos?)
├── Implementar sistema plug-and-play
└── Ablation: ¿cuántos slots son necesarios? ¿2? ¿4? ¿8?

Entregable: modelo MoE con fine-tune modular validado
Paper potencial: "Modular Expert Slots: Plug-and-Play 
                  Fine-Tuning without Catastrophic Forgetting"
```

### Fase 2 — Recurrencia Adaptativa (Meses 4-6)

```
Objetivo: añadir recurrencia al Bloque B con halting aprendido

Pasos:
├── Modificar Bloque B para permitir recurrencia
│   (weight sharing entre iteraciones)
├── Implementar halt head con ACT/PonderNet
├── Implementar loss dual (accuracy + compute penalty)
├── Entrenar con curriculum:
│   - Empezar con problemas de dificultad variable conocida
│   - El modelo aprende a calibrar número de iteraciones
├── Implementar budget estimator (Controlador de Recursos)
├── Entrenar budget estimator con RL
├── Evaluar:
│   - Accuracy vs. número de iteraciones en diferentes benchmarks
│   - ¿Usa más iteraciones para problemas difíciles? (calibración)
│   - Comparar con CoT externo equivalente
│   - Overhead de recurrencia vs. beneficio
└── Ablation: max_steps = {2, 4, 8, 16}

Entregable: modelo con recurrencia adaptativa funcional
Resultado esperado: +5-15% en benchmarks de razonamiento 
                    con ~30% menos compute promedio
```

### Fase 3 — Capas Heterogéneas (Meses 5-7, paralelo con Fase 2)

```
Objetivo: validar que la heterogeneidad de capas mejora eficiencia

Pasos:
├── Implementar Bloque A con SSM (Mamba-2)
├── Implementar transición SSM → Transformer (Bloque A → B)
├── Comparar arquitecturas:
│   - Homogénea: todas las capas iguales (baseline)
│   - Heterogénea tipo 1: solo tamaños diferentes
│   - Heterogénea tipo 2: tamaños + tipos diferentes (SSM+Transformer)
│   - Heterogénea tipo 3: tipo 2 + MoE solo en bloque central
├── Evaluar:
│   - Accuracy vs. FLOPs para cada configuración
│   - Throughput de inferencia (tokens/segundo)
│   - Perplexity por parámetro activo
└── NAS ligero para encontrar configuración óptima

Entregable: configuración heterogénea óptima validada
```

### Fase 4 — Scratchpad Mutable (Meses 7-11)

```
Objetivo: implementar y entrenar el scratchpad completo

Pasos:
├── MES 7-8: Implementación
│   ├── Banco KV externo con operaciones CRUD
│   ├── Cross-attention desde Bloque B al scratchpad
│   ├── Heads especializados para operaciones
│   └── Sistema de versionado de slots
│
├── MES 8-9: Fase 1-2 de entrenamiento (Sintético + SFT)
│   ├── Generar dataset sintético con modelo frontera
│   ├── Curar y filtrar traces (solo los útiles)
│   ├── SFT sobre el modelo con traces de scratchpad
│   └── Validar que el modelo usa el scratchpad (aunque sea torpemente)
│
├── MES 9-10: Fase 3 de entrenamiento (RL)
│   ├── Definir reward function completa
│   ├── Entrenar con PPO/GRPO
│   ├── El modelo aprende CUÁNDO usar el scratchpad
│   └── Iterar sobre reward function según resultados
│
├── MES 10-11: Fase 4 (Self-play)
│   ├── Auto-generación de datos de entrenamiento
│   ├── Filtrado por calidad
│   ├── Re-entrenamiento iterativo (3-5 ciclos)
│   └── Evaluación final
│
├── Evaluar:
│   - ¿El scratchpad mejora accuracy en razonamiento multi-paso?
│   - ¿Cuántos problemas usan scratchpad vs. no? (debería ser selectivo)
│   - Análisis cualitativo: ¿qué escribe el modelo en el scratchpad?
│   - ¿Se autocorrige? (edita/borra hipótesis incorrectas)
│   - Comparación con CoT externo equivalente en tokens totales
└── Ablation: número de slots, dimensión, frecuencia de acceso

Entregable: scratchpad funcional y validado
Paper potencial: "Learning to Think Without Speaking: 
                  Mutable Scratchpads for Internal Reasoning in LLMs"
```

### Fase 5 — Coprocesadores y Memoria (Meses 10-14, parcialmente paralelo)

```
Objetivo: integrar herramientas inline y memoria episódica

Pasos:
├── Coprocesadores:
│   ├── Implementar dispatch layer con Gumbel-softmax
│   ├── Coprocesador aritmético (symbolic math engine)
│   ├── Coprocesador de lookup (base de conocimiento vectorial)
│   ├── Coprocesador de verificación lógica (SAT solver simple)
│   ├── Entrenamiento: RL sobre cuándo invocar cada coprocesador
│   └── Evaluar: mejora en GSM8K, MATH, factual QA
│
├── Memoria episódica:
│   ├── Implementar vector store con 3 tipos de memoria
│   ├── Memory retrieval layer con gating
│   ├── Mecanismo de escritura de nuevas memorias post-inferencia
│   ├── Evaluar: consistencia a lo largo de conversaciones
│   └── Evaluar: personalización (¿recuerda preferencias del usuario?)
│
└── Integración de ambos con el sistema completo

Entregable: sistema completo COGA funcional
```

### Fase 6 — Integración y Optimización (Meses 14-16)

```
Objetivo: integrar todos los componentes, optimizar, y evaluar el sistema completo

Pasos:
├── Integración completa de todos los componentes
├── Optimización de inferencia:
│   ├── Cuantización (INT8/INT4) del modelo
│   ├── Kernel fusion para operaciones de scratchpad
│   ├── Batching eficiente con recurrencia variable
│   └── KV cache compartido entre iteraciones de recurrencia
├── Evaluación final exhaustiva (ver sección 5: Métricas)
├── Comparación con SOTA:
│   - Llama 3.x 70B
│   - Mixtral 8x22B
│   - GPT-4o-mini
│   - DeepSeek-R1-distilled
├── Análisis de eficiencia:
│   - FLOPs por query (distribución, no promedio)
│   - Tokens/segundo en hardware estándar
│   - Coste por query vs. modelos equivalentes
└── Documentación completa y release

Entregable: COGA v1.0 funcional y evaluado
Timeline total: ~16 meses
```

### Timeline visual

```
Mes:  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16
      ├──┤                                                     Fase 0: Fundamentos
         ├────────┤                                            Fase 1: Expert Slots
                  ├────────┤                                   Fase 2: Recurrencia
               ├──────────┤                                    Fase 3: Heterogénea
                           ├──────────────────┤                Fase 4: Scratchpad
                                    ├──────────────────┤       Fase 5: Tools+Memoria
                                                   ├──────┤   Fase 6: Integración
Papers:     ──────────────►P1                 ──────►P2    ──►P3
```

---

## 4. Experimentos Iniciales Sugeridos

### Experimento 1: Validación de Expert Slots (Quick Win)

```
Hipótesis: Fine-tune modular con expert slots preserva 
           ≥99% de performance general mientras alcanza 
           ≥95% de performance de fine-tune completo en dominio.

Setup:
- Base: Mixtral 8x7B (ya es MoE, modificar para tener 2 slots vacíos)
- Alternativa barata: entrenar MoE pequeño desde cero (~1-2B activos)
- Dominio test: medicina (PubMedQA, MedQA, MedMCQA)

Comparaciones:
A) Mixtral base (sin fine-tune)          → baseline general
B) Mixtral full fine-tune en medicina    → techo de dominio, suelo de general
C) Mixtral LoRA fine-tune               → baseline de eficiencia
D) Mixtral Expert Slots (nuestra idea)  → target

Métricas:
- MedQA accuracy (dominio)
- MMLU accuracy (general)
- MMLU-medicina (intersección)
- Ratio: accuracy_dominio / accuracy_general_loss

GPU budget estimado: ~8x A100 × 1 semana
```

### Experimento 2: Recurrencia vs. Profundidad

```
Hipótesis: Un modelo de 12 capas con recurrencia x4 iguala 
           o supera a un modelo de 48 capas sin recurrencia, 
           usando menos parámetros.

Setup:
- Modelo A: Transformer 48 capas, d=2048 (~6B params)
- Modelo B: Transformer 12 capas, d=2048, recurrencia max=8 (~1.5B params)
- Modelo C: Transformer 12 capas, d=4096, sin recurrencia (~6B params, 
            mismos FLOPs pero más anchos)
- Todos entrenados con el mismo dataset (15-20B tokens, alta calidad)

Evaluación:
- GSM8K (razonamiento matemático multi-paso)
- ARC-Challenge (razonamiento científico)
- MATH (matemáticas avanzadas)
- BBH (Big-Bench Hard, razonamiento general)
- Medir: steps de recurrencia promedio por categoría de dificultad

Pregunta clave: ¿El modelo B aprende a usar MÁS iteraciones 
               para problemas más difíciles?

GPU budget: ~8x A100 × 2 semanas (3 modelos)
```

### Experimento 3: Scratchpad vs. Chain-of-Thought

```
Hipótesis: Un scratchpad mutable interno supera a CoT externo 
           en eficiencia (tokens totales) y al menos iguala en accuracy.

Setup:
- Modelo base: mismo modelo, ~3B parámetros
- Variante A: CoT externo (genera razonamiento visible)
- Variante B: Scratchpad interno (piensa sin emitir)
- Variante C: Ambos (scratchpad + puede emitir razonamiento)

Dataset de evaluación:
- Problemas de lógica con dificultad graduada
- Puzzles que requieren explorar y descartar hipótesis
- Problemas donde la primera intuición suele ser incorrecta

Métricas:
- Accuracy por nivel de dificultad
- Tokens totales generados (incluyendo CoT visible)
- Operaciones de scratchpad (para variante B)
- Tasa de autocorrección: ¿cuántas veces el modelo cambió de hipótesis?
- Latencia de respuesta

GPU budget: ~4x A100 × 3 semanas (incluyendo RL del scratchpad)
```

### Experimento 4: Capas Heterogéneas (Ablation Rápida)

```
Hipótesis: La distribución no-uniforme de parámetros entre 
           capas mejora la ratio accuracy/FLOP.

Setup (modelos pequeños para iteración rápida):
- Todos ~1B parámetros totales, 24 capas

Configuraciones:
A) Homogéneo:    todas las capas d=1024
B) Piramidal:    d crece 512→1536 y vuelve a 512
C) Centro pesado: capas 1-6: d=512, capas 7-18: d=1536, capas 19-24: d=512
D) Inicio pesado: capas 1-8: d=1536, resto: d=768
E) Híbrido tipo: capas 1-6: Mamba d=512, capas 7-18: Transformer d=1536, 
                 capas 19-24: Transformer d=512

Entrenamiento: mismo dataset, mismos tokens, mismos FLOPs totales
Evaluación: perplexity + benchmarks downstream

GPU budget: ~4x A100 × 1 semana (modelos pequeños)
```

### Experimento 5: Budget Estimator — ¿Sabe el modelo qué es difícil?

```
Hipótesis: Un head lightweight puede predecir la dificultad 
           de un query con correlación significativa con el 
           número óptimo de pasos de razonamiento.

Setup (barato, puede hacerse con modelos existentes):
1. Tomar un modelo existente (e.g. Llama 3.1 8B)
2. Generar 10,000 problemas de dificultad variable
3. Para cada problema, medir "dificultad real":
   - ¿Cuántos tokens de CoT necesitó para acertar?
   - ¿Acertó en el primer intento o necesitó varios?
4. Entrenar un head linear sobre las activaciones de la capa 1
   para predecir la dificultad real
5. Medir correlación

Pregunta: ¿Las representaciones iniciales ya contienen 
          información sobre la dificultad del problema?

GPU budget: ~1x A100 × 2 días (muy barato)
```

---

## 5. Métricas de Evaluación

### 5.1 Métricas de Capacidad

```
┌──────────────────────────────────────────────────────────┐
│                 MÉTRICAS DE CAPACIDAD                    │
│                                                          │
│  Razonamiento:                                           │
│  ├── GSM8K (math word problems)                          │
│  ├── MATH (competition mathematics)                      │
│  ├── ARC-Challenge (science reasoning)                   │
│  ├── BBH (Big-Bench Hard, 27 tasks)                      │
│  ├── GPQA (graduate-level QA)                            │
│  └── MuSR (multi-step reasoning, custom)                 │
│                                                          │
│  Conocimiento:                                           │
│  ├── MMLU (57 subjects)                                  │
│  ├── MMLU-Pro (harder version)                           │
│  ├── TriviaQA                                            │
│  └── Domain-specific (MedQA, LegalBench, etc.)           │
│                                                          │
│  Código:                                                 │
│  ├── HumanEval / HumanEval+                              │
│  ├── MBPP / MBPP+                                        │
│  └── SWE-bench (real-world coding)                       │
│                                                          │
│  Lenguaje general:                                       │
│  ├── HellaSwag                                           │
│  ├── WinoGrande                                          │
│  └── PIQA                                                │
│                                                          │
│  Instrucciones/Chat:                                     │
│  ├── MT-Bench                                            │
│  ├── AlpacaEval 2.0                                      │
│  └── Arena-Hard                                          │
└──────────────────────────────────────────────────────────┘
```

### 5.2 Métricas de Eficiencia (las más importantes para nosotros)

```
┌──────────────────────────────────────────────────────────┐
│               MÉTRICAS DE EFICIENCIA                     │
│                                                          │
│  Compute:                                                │
│  ├── FLOPs por query (distribución, no solo media)       │
│  ├── FLOPs por token generado                            │
│  ├── Ratio: accuracy / GFLOPs (nuestra métrica clave)    │
│  ├── Tokens por segundo en hardware fijo                 │
│  └── Time-to-first-token latency                         │
│                                                          │
│  Parámetros:                                             │
│  ├── Parámetros totales vs. parámetros activos           │
│  ├── Ratio: accuracy / parámetros activos                │
│  └── Memoria GPU requerida (inference)                   │
│                                                          │
│  Adaptividad:                                            │
│  ├── Correlación(dificultad_real, compute_usado)         │
│  ├── Varianza del compute entre queries                  │
│  │   (alta varianza = buena adaptación)                  │
│  └── Overhead de compute adaptativo vs. fijo             │
└──────────────────────────────────────────────────────────┘
```

### 5.3 Métricas específicas de COGA

```
┌──────────────────────────────────────────────────────────┐
│              MÉTRICAS ESPECÍFICAS COGA                   │
│                                                          │
│  Scratchpad:                                             │
│  ├── Tasa de uso: % queries que activan scratchpad       │
│  ├── Tasa de autocorrección: % veces que EDIT/DELETE     │
│  │   una hipótesis incorrecta                            │
│  ├── Eficiencia: ops de scratchpad / mejora en accuracy  │
│  ├── Selectividad: ¿lo usa solo cuando es útil?          │
│  │   (correlación con dificultad)                        │
│  └── Análisis cualitativo: ¿qué escribe? ¿tiene sentido? │
│                                                          │
│  Recurrencia:                                            │
│  ├── Distribución de steps por categoría de problema     │
│  ├── Calibración: ¿más steps = más difícil?              │
│  ├── Convergencia: ¿cuántas veces hit hard cap?          │
│  ├── Marginal value: accuracy gain por step adicional    │
│  └── Efficiency frontier: accuracy vs. steps (Pareto)    │
│                                                          │
│  Expert Slots:                                           │
│  ├── Retención: accuracy general post fine-tune          │
│  ├── Especialización: accuracy en dominio                │
│  ├── Router accuracy: % queries correctamente ruteadas   │
│  ├── Composabilidad: accuracy con múltiples experts      │
│  │   cargados simultáneamente                            │
│  └── Swap time: latencia de cargar/descargar expert      │
│                                                          │
│  Coprocesadores:                                         │
│  ├── Tasa de invocación correcta: usa calc cuando debe   │
│  ├── Tasa de falso positivo: usa calc cuando no debe     │
│  ├── Accuracy improvement: con vs. sin coprocesadores    │
│  └── Latency overhead de dispatch                        │
│                                                          │
│  Budget Estimator:                                       │
│  ├── Correlación dificultad estimada vs. real            │
│  ├── Calibración: ¿subestima? ¿sobreestima?              │
│  ├── Compute savings vs. modelo con budget fijo          │
│  └── Failure rate: % veces que budget insuficiente       │
│     causó respuesta incorrecta                           │
└──────────────────────────────────────────────────────────┘
```

### 5.4 Métrica compuesta principal

```python
def COGA_score(model, benchmark_suite):
    """
    Métrica compuesta que captura la esencia de nuestro objetivo:
    máxima inteligencia por unidad de compute.
    """
    accuracy = evaluate_accuracy(model, benchmark_suite)
    avg_flops = measure_average_flops(model, benchmark_suite)
    active_params = count_active_parameters(model)
    
    # Intelligence Efficiency Ratio (IER)
    # Cuanto mayor, mejor: más accuracy por menos compute
    IER = accuracy / log(avg_flops)
    
    # Parameter Efficiency Ratio (PER)
    PER = accuracy / log(active_params)
    
    # Adaptivity Score: ¿el modelo adapta su compute?
    flops_per_query = [measure_flops(model, q) for q in benchmark_suite]
    difficulty_per_query = [estimate_difficulty(q) for q in benchmark_suite]
    adaptivity = spearman_correlation(flops_per_query, difficulty_per_query)
    
    # COGA Score compuesto
    COGA = (0.4 * IER + 0.3 * PER + 0.3 * adaptivity)
    
    return {
        'COGA_score': COGA,
        'IER': IER,
        'PER': PER,
        'adaptivity': adaptivity,
        'raw_accuracy': accuracy,
        'avg_flops': avg_flops,
        'active_params': active_params
    }
```

---

## 6. Desafíos Anticipados y Soluciones

### Desafío 1: Entrenamiento del Scratchpad — Señal de reward sparse

```
PROBLEMA:
El scratchpad opera internamente. El reward solo viene al 
final (respuesta correcta/incorrecta). Con muchas operaciones 
de scratchpad entre input y output, el gradiente se diluye.
¿Cómo sabe el modelo que la operación WRITE en el paso 3 
fue la que le llevó a la respuesta correcta?

SEVERIDAD: 🔴 Alta

SOLUCIONES:

1. Reward shaping intermedio
   - No solo premiar la respuesta final
   - Premiar operaciones intermedias verificables:
     * WRITE una hipótesis que luego resulta correcta → +reward
     * DELETE una hipótesis incorrecta → +reward
     * Uso de coprocesador que da resultado correcto → +reward
   - Requiere un "verifier" que pueda evaluar pasos intermedios

2. Hindsight relabeling
   - Después de resolver un problema, analizar retrospectivamente
     qué operaciones de scratchpad fueron útiles
   - Relablear esas operaciones como "buenas" para SFT adicional

3. Curriculum de complejidad de scratchpad
   - Empezar con problemas que requieren solo 1-2 operaciones
   - Gradualmente aumentar a problemas que requieren 5-10
   - El modelo aprende el valor de cada operación incrementalmente

4. Auxiliary losses en cada operación
   - Cada WRITE predice: "¿esto será útil?" → se verifica post-hoc
   - Loss auxiliar: predicción de utilidad de cada operación
   - Actúa como "credit assignment" explícito

SOLUCIÓN PREFERIDA: Combinación de 1 + 3
Empezar con curriculum simple + reward shaping intermedio.
Añadir hindsight relabeling en la fase de self-play.
```

### Desafío 2: Recurrencia — Estabilidad del entrenamiento

```
PROBLEMA:
Reusar los mismos pesos N veces amplifica gradientes.
Si N es grande, puede haber vanishing/exploding gradients
igual que en RNNs clásicas. Además, el modelo podría 
aprender a "nunca parar" para maximizar accuracy sin importar 
el coste, o "siempre parar inmediatamente" para minimizar loss 
de compute.

SEVERIDAD: 🟡 Media (hay soluciones conocidas parcialmente)

SOLUCIONES:

1. Gradient clipping por iteración
   - Clipear gradientes independientemente en cada paso
     de recurrencia, no solo al final
   
2. Detach parcial
   - Cada K iteraciones, detach el grafo computacional
   - Limita la profundidad del backprop a K steps
   - Trade-off: menos vanishing pero gradientes menos precisos

3. Regularización de la halt probability
   - Añadir entropy bonus al halt head
   - Evita colapso a "siempre parar" o "nunca parar"
   - halt_loss = -H(halt_distribution) * β

4. Warm-up de recurrencia
   - Empezar entrenamiento con max_steps = 1 (sin recurrencia)
   - Gradualmente aumentar max_steps durante el entrenamiento
   - El modelo aprende primero a funcionar en 1 paso
   - Luego aprende que más pasos pueden ayudar

5. LayerNorm entre iteraciones
   - Normalizar activaciones entre cada iteración de recurrencia
   - Previene drift de magnitud

6. Skip connections entre iteraciones
   - output_step_n = f(input_step_n) + input_step_n * α
   - Garantiza flujo de gradientes

SOLUCIÓN PREFERIDA: 4 + 5 + 6
Warm-up de recurrencia (probado en Universal Transformer) 
+ LayerNorm + skip connections residuales entre iteraciones.
```

### Desafío 3: Expert Slots — Router collapse durante fine-tune

```
PROBLEMA:
Al entrenar el nuevo experto + router, el router podría:
a) Enviar TODO al nuevo experto (ignora los base) → pierde generalidad
b) NUNCA enviar al nuevo experto (inercia) → no aprende el dominio
c) Routing inconsistente entre capas (experto 13 en capa 5, 
   experto 7 en capa 10 para la misma query médica)

SEVERIDAD: 🟡 Media

SOLUCIONES:

1. Load balancing loss (ya existe en MoE estándar)
   L_balance = α * CV(expert_usage)²
   Penaliza distribución muy desigual de uso entre expertos

2. Routing consistency loss (nuevo)
   L_consistency = variance(routing_decisions_across_layers)
   Si la capa 5 envía a expert 13, la capa 10 debería también
   (para queries del mismo dominio)

3. Mixto obligatorio durante fine-tune (tu idea)
   - 40% datos de dominio: aquí el router APRENDE a derivar
   - 40% datos generales: aquí el router MANTIENE routing existente
   - 20% adversarial: aquí el router aprende boundaries
   - Este approach previene a) y b) simultáneamente

4. Router warm-up separado
   - Fase 1: entrenar solo expert 13 con routing FORZADO
   - Fase 2: descongelar router y calibrar con datos mixtos
   - Evita que el router interfiera con el aprendizaje del experto

5. Expert prototype vectors
   - Cada expert slot tiene un "vector de identidad" que describe
     su dominio (inicializado con embedding del nombre del dominio)
   - El router compara query embedding con prototipos
   - Más interpretable y controlable

SOLUCIÓN PREFERIDA: 3 + 4
Datos mixtos (tu propuesta, que es excelente) + warm-up del 
router en dos fases. Es simple, robusto y controlable.
```

### Desafío 4: Coprocesadores — Diferenciabilidad

```
PROBLEMA:
Las herramientas externas (calculadora, DB lookup) no son 
diferenciables. No podemos backpropagar a través de calc(2+2)=4.
¿Cómo entrena el modelo cuándo invocar una herramienta?

SEVERIDAD: 🟡 Media

SOLUCIONES:

1. Straight-Through Estimator (STE)
   - Forward: decisión discreta (usar/no usar herramienta)
   - Backward: gradiente pasa como si fuera continuo
   - Simple, funciona razonablemente bien

2. Gumbel-Softmax para la decisión
   - Relaja la decisión discreta a continua durante training
   - τ (temperatura) se reduce gradualmente → decisiones más discretas

3. REINFORCE / Policy Gradient para la decisión de dispatch
   - Tratar la invocación de herramientas como una acción de RL
   - Reward: ¿la herramienta mejoró la respuesta?
   - No requiere diferenciabilidad

4. Proxy diferenciable
   - Entrenar una red pequeña que APROXIME cada herramienta
   - Training: usar la proxy (diferenciable)
   - Inference: usar la herramienta real (precisa)
   - Trade-off: la proxy introduce error de aproximación en training

5. Two-phase training
   - Fase 1 (SFT): entrenar con traces que incluyen uso de herramientas
     (el modelo aprende CUÁNDO por imitación, no por gradientes)
   - Fase 2 (RL): refinar con reward real
   - No requiere diferenciabilidad en ningún momento

SOLUCIÓN PREFERIDA: 5 (SFT + RL) con 2 (Gumbel-Softmax) como complemento.
Más robusto que depender solo de trucos de gradientes.
```

### Desafío 5: Complejidad de ingeniería — Integración de todo

```
PROBLEMA:
Cada componente por separado es manejable. Todos juntos crean 
una complejidad de sistema enorme:
- ¿Cómo hacer batching eficiente cuando cada query tiene 
  diferente número de iteraciones de recurrencia?
- ¿Cómo manejar el scratchpad en un batch?
- ¿Los coprocesadores son bottleneck si muchas queries 
  necesitan herramientas simultáneamente?
- Debugging: si algo falla, ¿dónde está el problema?

SEVERIDAD: 🟠 Media-Alta (riesgo de proyecto)

SOLUCIONES:

1. Desarrollo incremental estricto (nuestro plan de fases)
   - Cada fase añade UN componente
   - Validar que funciona antes de añadir el siguiente
   - Si un componente no aporta, se descarta

2. Batching con padding inteligente
   - Agrupar queries por dificultad estimada (similar budget)
   - Queries simples: batch grande, pocas iteraciones
   - Queries complejas: batch pequeño, muchas iteraciones
   - Variante: continuous batching como en vLLM

3. Scratchpad per-query isolation
   - Cada query en el batch tiene su propio banco de scratchpad
   - No interfieren entre sí
   - Implementación: dimensión de batch separada en el banco KV

4. Coprocesador pool asíncrono
   - Pool de workers para herramientas
   - Si una query necesita calc(), el dispatch es asíncrono
   - Las queries que no necesitan herramientas no se bloquean

5. Observabilidad exhaustiva
   - Dashboard en tiempo real:
     * Uso de scratchpad por query
     * Iteraciones de recurrencia
     * Routing de expertos
     * Invocaciones de herramientas
     * Budget estimado vs. real
   - Logging estructurado de CADA decisión interna
   - Reproducibilidad: seeds + traces completos

SOLUCIÓN PREFERIDA: Todas. Es un riesgo de ingeniería, 
no de investigación. Se mitiga con buenas prácticas de 
desarrollo, no con ideas brillantes.
```

### Desafío 6: Evaluación — ¿Cómo sabemos que funciona "por las razones correctas"?

```
PROBLEMA:
Un modelo podría obtener buenos benchmarks sin realmente usar 
el scratchpad/recurrencia/herramientas de forma significativa.
¿Está el scratchpad contribuyendo o es teatro?

SEVERIDAD: 🟡 Media

SOLUCIONES:

1. Ablation studies rigurosos
   - Modelo completo vs. sin scratchpad vs. sin recurrencia
   - Si quitar un componente no cambia accuracy, sobra

2. Causal interventions
   - Corromper el scratchpad mid-inference: ¿empeora la respuesta?
   - Forzar más/menos iteraciones: ¿cambia la calidad?
   - Bloquear herramientas en problemas que las necesitan

3. Análisis cualitativo del scratchpad
   - Humanos leen el scratchpad y evalúan:
     * ¿El contenido es coherente?
     * ¿Las correcciones son genuinas?
     * ¿Se parece a razonamiento humano?
   
4. Problemas diseñados para requerir cada componente
   - Problemas que SOLO se resuelven con exploración (→ scratchpad)
   - Problemas que SOLO se resuelven con más pasos (→ recurrencia)
   - Problemas que SOLO se resuelven con cálculo exacto (→ coprocesador)

SOLUCIÓN PREFERIDA: 1 + 4. Ablations + benchmarks diseñados 
para aislar la contribución de cada componente.
```

### Resumen de riesgos

```
┌──────────────────────────┬───────────┬────────────────────┐
│ Desafío                  │ Severidad │ Mitigación         │
├──────────────────────────┼───────────┼────────────────────┤
│ Reward sparse scratchpad │ 🔴 Alta   │ Reward shaping +   │
│                          │           │ curriculum         │
│ Estabilidad recurrencia  │ 🟡 Media  │ Warm-up + LayerNorm│
│ Router collapse          │ 🟡 Media  │ Datos mixtos +     │
│                          │           │ warm-up router     │
│ Diferenciabilidad tools  │ 🟡 Media  │ SFT + RL           │
│ Complejidad ingeniería   │ 🟠 Med-Alt│ Fases incrementales│
│ Evaluación significativa │ 🟡 Media  │ Ablations + tests  │
│                          │           │ causales           │
└──────────────────────────┴───────────┴────────────────────┘
```

---

## Apéndice A: Equipo y recursos estimados

```
Equipo mínimo:
├── 2-3 ML researchers (arquitectura + entrenamiento)
├── 1-2 ML engineers (infraestructura + optimización)
├── 1 research engineer (evaluación + benchmarks)

Compute estimado (total del proyecto):
├── Fase 0-1: ~500 GPU-hours A100
├── Fase 2-3: ~2,000 GPU-hours A100
├── Fase 4:   ~5,000 GPU-hours A100
├── Fase 5-6: ~3,000 GPU-hours A100
├── Total:    ~10,000-15,000 GPU-hours A100
│             (~$150K-$250K en cloud compute)
│             (accesible para lab universitario o startup)

Comparación: entrenar GPT-4 costó estimadamente $100M+
            Nuestro proyecto es ~500-1000x más barato
```

## Apéndice B: Trabajo relacionado clave

```
Scratchpad/Memoria de trabajo:
- Neural Turing Machine (Graves et al., 2014)
- Differentiable Neural Computer (Graves et al., 2016)
- Memory Networks (Weston et al., 2015)
- Memorizing Transformers (Wu et al., 2022)

Recurrencia adaptativa:
- Adaptive Computation Time (Graves, 2016)
- Universal Transformer (Dehghani et al., 2019)
- PonderNet (Banino et al., 2021)

MoE y routing:
- Switch Transformer (Fedus et al., 2022)
- Mixtral (Jiang et al., 2024)
- DeepSeek-MoE (Dai et al., 2024)

Tool use:
- Toolformer (Schick et al., 2023)
- Gorilla (Patil et al., 2023)

Arquitecturas eficientes:
- Mamba (Gu & Dao, 2024)
- Jamba (Lieber et al., 2024)
- Phi-3 (Abdin et al., 2024)

Test-time compute:
- OpenAI o1 (2024)
- DeepSeek-R1 (2025)
```

---

Este es el research proposal completo. ¿Quieres que ahora pasemos a los otros dos documentos?

1. **Paper sobre Modular Expert Slots** (la idea más publicable)
2. **Implicaciones filosóficas del Scratchpad** (pensamiento sin emisión, experiencia interna, aumento de capacidades)


---
