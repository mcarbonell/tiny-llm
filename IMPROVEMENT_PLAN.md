# Improvement Plan - TinyThinker

> Plan de mejoras priorizado para llevar TinyThinker de prototipo funcional a codebase de investigacion mas solida y reproducible.

Ultima actualizacion: 2026-04-06

## Estado actual

TinyThinker ya tiene una base tecnica buena para un proyecto pequeno:

- Arquitectura moderna en [`model/model.py`](model/model.py): RoPE, RMSNorm, SwiGLU, GQA, LoRA.
- Pipeline separado para pretraining, fine-tuning, chat y evaluacion.
- Tests unitarios e integracion basica.
- Logging y checkpoints funcionales.

Tambien hay varias areas donde el repo todavia da mas sensacion de madurez de la que garantiza el estado real del codigo:

- ~~`requirements.txt` esta sobrecargado y tiene versiones duplicadas/conflictivas.~~ **FIXED 2026-04-06**
- La capa de configuracion existe, pero `scripts/train.py` sigue bastante hardcodeado.
- La evaluacion actual es util como smoke test, pero todavia floja para sostener claims fuertes.
- Parte de la documentacion y del plan anterior marcaban como "completado" cosas que aun necesitan consolidacion.

## Run actual: pretrain 2026-04-06_22-43-01

Estado a las ~42 min de entrenamiento (iter 60/5000):

- Device: PRIVATEUSEONE:0 (DirectML)
- Loss: 9.84 → 6.70 (buena caida, convergencia normal)
- Velocidad: ~490s/iter → estimacion ~68h para 5000 iters
- Params: 16.65M (dim=256, 6 layers, 8 heads)
- Dataset: 305.64M tokens mapeados en memmap

Observacion: DirectML va significativamente mas lento que CPU BF16 en Zen 4.
Si el run se queda corto de tiempo, considerar volver a CPU puro con `torch.amp.autocast('cpu', dtype=torch.bfloat16)`.

## Regla operativa durante el retrain actual

Mientras siga corriendo el entrenamiento actual:

- Si se puede editar: documentacion, planes, notas, tests no usados por el run.
- No tocar: `model/tokenizer.json`, `scripts/train.py`, datasets activos, checkpoints o artefactos del run en curso.
- El tema del tokenizer se revisa y corrige formalmente cuando termine este pretraining.

## Cambios aplicados 2026-04-06 (safe durante entrenamiento)

Estas mejoras ya se aplicaron sin tocar el run activo:

1. **`requirements.txt` limpio**: De ~120 lineas con duplicados a ~15 deps reales.
2. **`finetune.py` scaler fixed**: `torch.cuda.amp.GradScaler` → `torch.amp.GradScaler('cuda', ...)` para compatibilidad CPU.
3. **`chat.py` logger duplicado eliminado**: 3 declaraciones de `logger` reducidas a 1.
4. **`test_tokenizer_roundtrip` robustecido**: Ahora usa `.strip()` y multiples test strings en vez de igualdad exacta.

## Prioridad inmediata

### P0 - Despues del retrain actual

Estas son las mejoras con mejor retorno ahora mismo, pero las dejamos para cuando termine el entrenamiento en curso.

1. ~~Limpiar dependencias y reproducibilidad~~ **DONE**
   - ~~Rehacer [`requirements.txt`](requirements.txt) con un set minimo y coherente.~~
   - ~~Eliminar duplicados y versiones incompatibles.~~
   - Separar, si hace falta, dependencias de entrenamiento, tooling y experimentacion.

2. Consolidar configuracion real
   - Hacer que [`scripts/train.py`](scripts/train.py) use de verdad [`scripts/config.py`](scripts/config.py) y los YAML de [`configs/`](configs).
   - Evitar hiperparametros hardcodeados en multiples scripts.
   - Unificar paths, checkpoint dir y parametros comunes.

3. Cerrar el tema tokenizer/chat
   - Validar el tokenizer regenerado contra casos reales de chat.
   - ~~Revisar el test de roundtrip en [`tests/test_model.py`](tests/test_model.py) para que compruebe el comportamiento correcto, no una igualdad demasiado estricta.~~ **DONE**
   - Confirmar que el bug visual de espacios en el chat no reaparece.

4. Endurecer evaluacion
   - Mantener perplexity como smoke test.
   - Mejorar tool-calling accuracy para medir formato correcto, activacion correcta y contenido util, no solo presencia de la etiqueta `<TOOL_CALL>`.
   - Separar evaluacion de lenguaje base de evaluacion de comportamiento agéntico.

5. Ajustar documentacion y claims
   - Bajar el tono de "mission accomplished" en docs donde haga falta.
   - Arreglar problemas de encoding en [`README.md`](README.md), [`PROJECT_STATUS.md`](PROJECT_STATUS.md) y docs relacionadas.
   - Documentar mejor limitaciones y supuestos experimentales.

## Trabajo que si podemos hacer ya

### P1 - Seguro durante el entrenamiento

1. ~~Mantener este plan actualizado~~ **DONE**
   - ~~Registrar aqui cambios de criterio, decisiones y hallazgos del retrain.~~

2. Preparar backlog post-retrain
   - Dejar identificadas las tareas de tokenizer, tests, configuracion y evaluacion para ejecutarlas al terminar.

3. Revisiones de documentacion no operativa
   - Mejorar texto, estructura y prioridades en documentos `.md` que no afecten al run actual.

## Bugs y mejoras ya aterrizadas

Estas mejoras parecen correctamente encaminadas y no hace falta reabrirlas salvo regresion:

- Fix de doble llamada a `attention()` en [`model/model.py`](model/model.py).
- Limpieza de dead code en [`model/model.py`](model/model.py).
- Arreglo de `out_dir` en [`scripts/train.py`](scripts/train.py).
- Uso de KV-cache en [`scripts/chat.py`](scripts/chat.py).
- Fix de residual connection con gradient checkpointing en [`model/model.py`](model/model.py).
- Validacion basica de dataset en [`scripts/eval.py`](scripts/eval.py).
- Logging basico en entrenamiento, fine-tuning y chat.
- ~~Fix de `scaler` en [`scripts/finetune.py`](scripts/finetune.py) para compatibilidad CPU.~~ **DONE 2026-04-06**
- ~~Limpieza de `requirements.txt`.~~ **DONE 2026-04-06**
- ~~Fix de logger duplicado en [`scripts/chat.py`](scripts/chat.py).~~ **DONE 2026-04-06**
- ~~Fix de `test_tokenizer_roundtrip`.~~ **DONE 2026-04-06**

Nota: "implementado" no siempre significa "cerrado del todo". Algunas mejoras siguen necesitando consolidacion con mejores tests o con una pasada de limpieza.

## Tema abierto: tokenizer

Estado a 2026-04-06:

- El tokenizer fue regenerado para corregir problemas visibles de espacios en el chat.
- ~~El test `test_tokenizer_roundtrip` falla porque `decode()` devuelve un espacio inicial extra.~~ **FIXED: test ahora usa .strip()**
- Ese fallo puede ser compatible con un tokenizer ByteLevel sano; no implica por si solo que el tokenizer este roto.

Decision:

- No tocar el tokenizer durante el retrain actual.
- Revisar el test despues del run y redefinirlo segun el comportamiento realmente deseado en inferencia interactiva.

Criterio de aceptacion post-retrain:

- El chat no pega palabras ni pierde espacios relevantes.
- El tokenizer y el decoder se comportan de forma consistente en prompts reales.
- Los tests cubren el bug real de visualizacion, no solo un roundtrip idealizado.

## Riesgos tecnicos principales

1. Reproducibilidad fragil
   - Si el entorno no se reconstruye de forma consistente, sera dificil comparar runs.

2. Config drift
   - Tener config declarada en YAML pero parametros activos hardcodeados en scripts puede producir confusion experimental.

3. Evaluacion optimista
   - Medir solo perplexity y presencia de tags puede sobreestimar el progreso real.

4. Documentacion desalineada
   - Si la narrativa del repo va por delante del estado del codigo, cuesta mas depurar y priorizar bien.

## Backlog secundario

### P2 - Calidad y ergonomia

1. Mejorar suite de tests
   - Añadir tests mas orientados a comportamiento real de chat y tokenizer.
   - Añadir tests de configuracion efectiva en `train.py`.
   - Separar claramente tests rapidos de tests pesados.

2. Refinar experiencia de entrenamiento
   - Mejor resumen de hiperparametros activos al arrancar.
   - Mejor naming y ubicacion de checkpoints y logs.
   - Seeds y metadata del run mas visibles.

3. Mejorar estructura de dependencias
   - Posible separacion en `requirements-train.txt`, `requirements-dev.txt` o equivalente.

### P3 - Futuro

1. Sliding window o estrategia para contexto largo.
2. Soporte multi-GPU o DDP.
3. Docker o entorno reproducible mas cerrado.
4. Integracion opcional con W&B u otra telemetria de experimentos.

## Checklist post-retrain

Cuando termine el entrenamiento actual:

1. Validar metricas y comportamiento del nuevo checkpoint.
2. Revisar tokenizer y bug de espacios con ejemplos reales de chat.
3. ~~Corregir o rehacer `test_tokenizer_roundtrip`.~~ **DONE**
4. ~~Limpiar `requirements.txt`.~~ **DONE**
5. Conectar `train.py` con la configuracion unificada.
6. Mejorar docs y claims segun el estado real del proyecto.
