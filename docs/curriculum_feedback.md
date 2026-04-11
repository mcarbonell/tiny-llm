Te veo muy buen esquema. Está bien estructurado y tiene progresión clara. Voy a señalar mejoras concretas que te ayudarán a tener un dataset más robusto.

# 1. Mejoras en LEVEL_SPEC

## A. Faltan niveles superiores (lo mencioné antes)
Te recomiendo al menos dos niveles más:

```python
    5: {
        "name": "Adult Abstract",
        "audience": "young adults ages 17 to 20",
        "focus": "Multi-agent reasoning, Bayesian inference, complex optimization, and trade-off analysis.",
        "topics": [
            "multi-person negotiation with incomplete information",
            "Bayesian updating with new evidence",
            "resource allocation under multiple constraints",
            "identifying causal vs. correlational relationships",
            "analyzing incentives in game theory scenarios",
            "system failure diagnosis with multiple potential causes",
            "prisoner's dilemma and similar strategic situations",
            "evaluating arguments with statistical evidence"
        ]
    },
    6: {
        "name": "Expert Meta-Reasoning",
        "audience": "adults 21+",
        "focus": "Meta-cognition, counterfactual analysis, epistemology, and reasoning about reasoning.",
        "topics": [
            "identifying missing assumptions in an argument",
            "evaluating if sufficient information exists to solve a problem",
            "comparing strengths of competing explanations",
            "recognizing cognitive biases in reasoning patterns",
            "designing experiments to test hypotheses",
            "counterfactual impact analysis in complex systems",
            "reasoning under radical uncertainty (unknown unknowns)",
            "epistemic status assessment (how confident should we be?)"
        ]
    }
```

## B. Topics podrían ser más diversos por tipo de formato

Ahora mismo tus topics describen contenido, pero no el **formato** del problema. Ejemplo:

- "transitive logic" → podría ser: tabla, narrativa, diagrama
- "math word problems" → podría tener distintas superficies lingüísticas

Te sugiero **tags adicionales** que podrías incorporar:

```python
FORMAT_TAGS = {
    "narrative": "Presented as a short story",
    "fact_list": "Presented as bullet points or listed facts",
    "table_grid": "Information in table or grid form",
    "dialogue": "Characters talking and giving clues",
    "rule_set": "Explicit if-then rules",
    "pseudocode": "Simple code-like instructions",
    "scenario": "Hypothetical situation description",
    "puzzle": "Traditional logic puzzle format"
}
```

## C. Mezclar niveles de dificultad dentro del mismo nivel

Ahora mismo cada nivel tiene topics del mismo grado de dificultad. Pero en la realidad, un niño de 5-7 años enfrenta problemas fáciles, medios y algunos desafiantes.

Podrías **indicar dificultad relativa** dentro del nivel:

```python
# Dentro de cada topic, podrías tener una tupla
"topics": [
    ("simple addition or subtraction with toys", "easy"),
    ("matching items by multiple attributes", "medium"),
    ("ordering three objects by height or weight", "easy"),
    ("predicting simple consequences with two variables", "hard"),
]
```

# 2. Mejoras en el SYSTEM_PROMPT

Tu prompt está bien, pero falta especificar **restricciones estructurales** importantes:

## Problemas actuales
1. **"Detailed step-by-step thinking"** → podría generar reasoning demasiado largo para niveles bajos
2. **Falta especificar extensión máxima** → modelos pueden generar páginas enteras
3. **"Use language appropriate for a {spec['audience']}'s teacher"** → ambiguo

## Propuesta mejorada

```python
def get_system_prompt(level):
    spec = LEVEL_SPEC[level]
    
    # Configuración por nivel
    level_config = {
        0: {"think_length": "2-3 sentences", "vocab": "very simple words"},
        1: {"think_length": "3-4 sentences", "vocab": "simple everyday language"},
        2: {"think_length": "4-5 sentences", "vocab": "clear explanatory language"},
        3: {"think_length": "5-6 sentences", "vocab": "precise technical terms when needed"},
        4: {"think_length": "6-8 sentences", "vocab": "formal but accessible language"},
    }
    
    config = level_config.get(level, {"think_length": "6-10 sentences", "vocab": "formal but accessible language"})
    
    return f"""You are creating logic puzzles for {spec['audience']} to develop reasoning skills.

CRITICAL REQUIREMENTS:
1. Question must require genuine logical thinking, not just recall.
2. The <think> section must show the complete reasoning process that leads to the answer.
3. Use {config['vocab']}.
4. Make the thinking section about {config['think_length']}.

REQUIRED FORMAT:
Question: [The puzzle or riddle]
<think>[Step-by-step reasoning. Show how you combine clues, eliminate possibilities, or calculate.]</think>
Answer: [Final answer]

ADDITIONAL RULES:
- No markdown, no bold, no lists in the Question.
- The thinking must be self-contained; don't refer to external knowledge.
- If the puzzle involves multiple steps, show each step clearly.
- The answer should be concise but complete.
- For level {level}, focus on: {spec['focus']}
- Start immediately with 'Question:' without any introductory phrases.

EXAMPLE FORMAT:
Question: Three children are in line. Mia is before Leo. Leo is before Sam. Who is first?
<think>We know: 1) Mia is before Leo. 2) Leo is before Sam. 
From (1), Mia comes before Leo. From (2), Leo comes before Sam.
So the order must be: Mia, then Leo, then Sam.
Therefore, Mia is first.</think>
Answer: Mia
"""
```

# 3. Mejoras en get_sample()

## Problemas en tu implementación
1. Falta manejo de errores
2. No hay control de calidad de la respuesta
3. No hay validación de formato

## Implementación mejorada

```python
import re
import json
from typing import Dict, Optional

def get_sample(level: int, topic: str, retry_count: int = 3) -> Optional[Dict]:
    """
    Generate a single training example.
    
    Returns: dict with keys: question, thinking, answer, level, topic
    Returns None if generation fails after retries.
    """
    prompt = get_system_prompt(level)
    
    # Más específico en el user prompt
    user_content = f"""Create a logic puzzle about: {topic}

Level: {LEVEL_SPEC[level]['name']}
Audience: {LEVEL_SPEC[level]['audience']}
Focus: {LEVEL_SPEC[level]['focus']}

Make sure the puzzle genuinely requires logical reasoning as specified."""

    for attempt in range(retry_count):
        try:
            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content}
                ],
                "max_tokens": 800 if level <= 2 else 1200,
                "temperature": 0.7 if level <= 2 else 0.5,  # Menos variación en niveles altos
                "top_p": 0.9,
            }
            
            response = call_api(payload)  # Tu función para llamar a la API
            
            if not response:
                continue
                
            content = response["choices"][0]["message"]["content"]
            
            # Parsear la respuesta
            parsed = parse_response(content)
            
            if parsed and validate_example(parsed, level):
                parsed["level"] = level
                parsed["topic"] = topic
                parsed["metadata"] = {
                    "audience": LEVEL_SPEC[level]["audience"],
                    "focus": LEVEL_SPEC[level]["focus"],
                    "attempt": attempt + 1
                }
                return parsed
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for level {level}, topic {topic}: {e}")
            continue
    
    print(f"Failed to generate valid example after {retry_count} attempts: level {level}, topic {topic}")
    return None

def parse_response(text: str) -> Optional[Dict]:
    """Extract question, thinking, and answer from response."""
    # Patrones más robustos
    question_pattern = r"Question:\s*(.*?)(?=\s*<think>|\s*Answer:|\Z)"
    think_pattern = r"<think>(.*?)</think>"
    answer_pattern = r"Answer:\s*(.*?)(?=\s*Question:|\Z)"
    
    question_match = re.search(question_pattern, text, re.DOTALL)
    think_match = re.search(think_pattern, text, re.DOTALL)
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    
    if not (question_match and think_match and answer_match):
        # Fallback: buscar cualquier indicio
        lines = text.strip().split('\n')
        # Heurística simple
        if len(lines) >= 3:
            # Asumir formato: pregunta, thinking, respuesta
            return {
                "question": lines[0].replace("Question:", "").strip(),
                "thinking": lines[1].replace("<think>", "").replace("</think>", "").strip(),
                "answer": lines[2].replace("Answer:", "").strip()
            }
        return None
    
    return {
        "question": question_match.group(1).strip(),
        "thinking": think_match.group(1).strip(),
        "answer": answer_match.group(1).strip()
    }

def validate_example(example: Dict, level: int) -> bool:
    """Basic validation rules."""
    question = example["question"]
    thinking = example["thinking"]
    answer = example["answer"]
    
    # Longitudes mínimas
    if len(question) < 10:
        return False
    
    if len(thinking) < 20:
        return False
    
    if len(answer) < 1:
        return False
    
    # El thinking no debe ser idéntico a la pregunta
    if thinking.lower() in question.lower() or question.lower() in thinking.lower():
        return False
    
    # El thinking debe contener palabras de razonamiento
    reasoning_words = ["because", "therefore", "since", "so", "thus", "if", "then", 
                      "first", "second", "next", "finally", "means that", "implies"]
    
    if level >= 2:  # Para niveles 2+, requerir al menos una palabra de razonamiento
        if not any(word in thinking.lower() for word in reasoning_words):
            return False
    
    # La respuesta no debe estar vacía o ser trivial
    trivial_answers = ["yes", "no", "true", "false", "maybe"]
    if answer.lower() in trivial_answers and len(thinking) < 50:
        # Si la respuesta es trivial, el thinking debe ser sustancial
        return len(thinking) > 80
    
    return True

def call_api(payload):
    """Mock - reemplazar con tu llamada real a la API"""
    # Implementación real aquí
    pass
```

# 4. Mejora crítica: Dataset balanceado

## Generar por habilidades, no solo por topic

Te sugiero un generador que asegure cobertura balanceada:

```python
REASONING_SKILLS = {
    "classification": ["sorting", "categorization", "odd-one-out"],
    "ordering": ["temporal", "spatial", "ranking"],
    "matching": ["one-to-one", "attributes", "ownership"],
    "arithmetic": ["addition", "subtraction", "multiplication", "division"],
    "logic": ["transitive", "negation", "conditional", "contradiction"],
    "spatial": ["positions", "paths", "arrangements"],
    "temporal": ["schedules", "sequences", "durations"],
    "causal": ["cause-effect", "predictions", "explanations"],
    "probabilistic": ["chance", "likelihood", "uncertainty"],
    "strategic": ["optimization", "trade-offs", "decisions"]
}

def generate_balanced_dataset(num_per_level=100):
    """Generate balanced dataset covering all skills."""
    dataset = []
    
    for level in LEVEL_SPEC:
        # Para cada nivel, generar ejemplos que cubran diferentes skills
        skills_for_level = get_skills_for_level(level)
        
        for skill in skills_for_level:
            for _ in range(num_per_level // len(skills_for_level)):
                # Elegir topic que ejercite esa skill
                topic = select_topic_for_skill(level, skill)
                if not topic:
                    continue
                    
                example = get_sample(level, topic)
                if example:
                    example["skill"] = skill
                    dataset.append(example)
    
    return dataset

def get_skills_for_level(level):
    """What skills are appropriate for each level."""
    skills_by_level = {
        0: ["classification", "ordering", "matching"],
        1: ["classification", "ordering", "matching", "arithmetic", "spatial"],
        2: ["ordering", "logic", "temporal", "causal", "arithmetic"],
        3: ["logic", "temporal", "causal", "probabilistic", "strategic"],
        4: ["logic", "probabilistic", "strategic", "causal"],
        5: ["strategic", "probabilistic", "logic", "causal"],
        6: ["strategic", "probabilistic", "logic"]  # Meta-skills
    }
    return skills_by_level.get(level, [])
```

# 5. Cambios inmediatos que te recomiendo

De tu código actual, esto es lo que cambiaría **ya**:

## 1. Añade niveles 5 y 6
Como mínimo, añade el nivel 5 para cubrir razonamiento adulto.

## 2. Mejora el prompt
Usa mi versión mejorada con configuraciones específicas por nivel.

## 3. Añade validación
No confíes en que el modelo siempre dé el formato correcto. Añade `parse_response()` y `validate_example()`.

## 4. Añade balance por skill
No solo generes por topic; asegúrate de cubrir diferentes habilidades de razonamiento.

## 5. Control de calidad
Para cada ejemplo generado, puedes añadir:
- Dificultad estimada (basada en longitud, número de entidades, etc.)
- Tipo de formato (narrativa, tabla, etc.)
- Habilidad principal ejercitada

# 6. Script completo mejorado

Te dejo una versión condensada con las mejoras clave:

```python
# Level Configuration con 7 niveles
LEVEL_SPEC = {
    0: {
        "name": "Foundation",
        "audience": "children ages 3 to 4",
        "focus": "Basic categorization, colors, sizes, and counting up to 5.",
        "topics": [
            "identifying the color of common fruits",
            "sorting animals by size (big vs small)",
            "counting items in a basket (up to 5)",
            "identifying which item doesn't belong (e.g., fruit vs toy)",
            "basic relationships: inside or outside a box",
            "simple colors: what is the color of the sky vs the sun",
            "matching identical objects",
            "simple patterns: red, blue, red, blue, what comes next?",
            "which animal makes which sound",
            "what do you wear on different body parts"
        ],
        "think_length": "2-3 sentences",
        "vocab": "very simple words"
    },
    # ... niveles 1-4 similares pero con más topics y configs ...
    4: {
        "name": "Formal Operations",
        "audience": "teenagers ages 14 to 16",
        "focus": "Abstraction, counterfactual reasoning, simple optimization, and systems.",
        "topics": [
            "counterfactual thinking (what if a premise was false?)",
            "simple optimization (shortest path through 4 points)",
            "systems of equations in word form",
            "evaluating the strength of a logical argument",
            "basic probability and risk assessment",
            "understanding basic trade-offs: cost vs speed",
            "conditional probability scenarios",
            "simple game theory decisions",
            "identifying logical fallacies in arguments",
            "resource allocation with constraints"
        ],
        "think_length": "6-8 sentences",
        "vocab": "formal but accessible language"
    },
    5: {
        "name": "Adult Abstract",
        "audience": "young adults ages 17 to 20",
        "focus": "Multi-agent reasoning, Bayesian inference, complex optimization, and trade-off analysis.",
        "topics": [
            "multi-person negotiation with incomplete information",
            "Bayesian updating with new evidence",
            "resource allocation under multiple constraints",
            "identifying causal vs. correlational relationships",
            "analyzing incentives in game theory scenarios",
            "system failure diagnosis with multiple potential causes",
            "prisoner's dilemma and similar strategic situations",
            "evaluating arguments with statistical evidence"
        ],
        "think_length": "8-12 sentences",
        "vocab": "precise technical language when needed"
    },
    6: {
        "name": "Expert Meta-Reasoning", 
        "audience": "adults 21+",
        "focus": "Meta-cognition, counterfactual analysis, epistemology, and reasoning about reasoning.",
        "topics": [
            "identifying missing assumptions in an argument",
            "evaluating if sufficient information exists to solve a problem",
            "comparing strengths of competing explanations",
            "recognizing cognitive biases in reasoning patterns",
            "designing experiments to test hypotheses",
            "counterfactual impact analysis in complex systems",
            "reasoning under radical uncertainty (unknown unknowns)",
            "epistemic status assessment (how confident should we be?)"
        ],
        "think_length": "10-15 sentences",
        "vocab": "precise technical language"
    }
}

def get_system_prompt(level):
    spec = LEVEL_SPEC[level]
    return f"""Create a logic puzzle for {spec['audience']} that develops reasoning skills.

IMPORTANT:
- Question must require genuine logical thinking
- Show COMPLETE reasoning in <think> section
- Use {spec.get('vocab', 'clear language')}
- Thinking should be {spec.get('think_length', 'appropriate length')}

FORMAT:
Question: [The puzzle]
<think>[Step-by-step reasoning. Show how clues combine or calculations work.]</think>
Answer: [Final answer]

RULES:
- No markdown in Question
- Thinking must be self-contained
- Show each logical step clearly
- Focus on: {spec['focus']}
- Start with 'Question:' immediately
- Make it challenging but solvable for {spec['audience']}

Example of good thinking for a simple puzzle:
<think>We know: 1) Anna is taller than Ben. 2) Ben is taller than Cara.
From (1), Anna > Ben. From (2), Ben > Cara.
So: Anna > Ben > Cara.
Therefore, Anna is tallest.</think>
"""

# Resto del código con las funciones mejoradas...
```

---

**Resumen de cambios recomendados:**

1. **Añade 2 niveles más** (5 y 6) para razonamiento adulto/experto
2. **Expande topics** a ~10 por nivel para más variedad
3. **Mejora el system prompt** con configs específicas por nivel
4. **Añade funciones de validación** y parsing robusto
5. **Considera balance por habilidad**, no solo por topic
6. **Añade metadata** para análisis posterior (skill, formato, dificultad)

¿Quieres que desarrolle alguno de estos componentes con más detalle?