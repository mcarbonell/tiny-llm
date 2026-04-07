# 📜 Estándares de Logging - Proyecto TinyThinker

Este documento define el formato obligatorio para todos los logs generados por los scripts de entrenamiento, fine-tuning y evaluación del proyecto. El objetivo es permitir una comparación técnica precisa entre diferentes ejecuciones.

## 1. Estructura del Archivo de Log
Cada sesión de ejecución debe generar un archivo único en la carpeta `logs/` con el prefijo del script y el timestamp: `scripts/train_YYYYMMDD_HHMMSS.log`.

## 2. Cabecera de Metadatos (Header)
Toda ejecución debe comenzar imprimiendo un bloque de metadatos delimitado por líneas de `=` que incluya:
- `DATE`: Fecha y hora de inicio del proceso.
- `DEVICE`: Dispositivo utilizado (`CPU`, `PRIVATEUSEONE:0`, `CUDA`, etc.).
- `CPU THREADS`: Número de hilos activos si se usa CPU.
- `--------------- HYPERPARAMS -----------`: Bloque con `batch_size`, `seq_len`, `grad_accum_steps`, `max_iters`, `learning_rate`.
- `--------------- MODEL PARAMS ----------`: Bloque con `dim`, `n_layers`, `n_heads`, `vocab_size` y `TOTAL PARAMS`.

## 3. Formato de las Líneas de Log
Cada línea de log debe seguir estrictamente este formato:
`[HH:MM:SS] <mensaje>`

Donde **`[HH:MM:SS]` NO es la hora actual**, sino el **tiempo transcurrido** desde el inicio de la ejecución (`elapsed time`). Esto facilita comparar la velocidad de convergencia y el rendimiento por hora de diferentes configuraciones.

### Ejemplo de línea de entrenamiento:
`[00:15:30] iter   250 | loss 3.1245 | lr 1.00e-03 | time 7.45s`

## 4. Unidades de Medida
- **Tiempos de bucle (`time`)**: Siempre en segundos (`s`), con dos decimales. Evitar milisegundos (`ms`) para mayor legibilidad.
- **Pérdida (`loss`)**: Siempre con cuatro decimales.
- **Learning Rate (`lr`)**: Siempre en notación científica (`1.00e-03`).

## 5. Caracteres Especiales
- **No usar Emojis**: Para garantizar la compatibilidad con todas las terminales de Windows (encoding CP1252), se prohíbe el uso de emojis en los archivos de log.
- **Separadores**: Usar la barra vertical `|` para separar métricas en una misma línea.
