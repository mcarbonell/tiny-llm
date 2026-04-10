# ⚡ Análisis de Cómputo: Modal vs Local (Radeon 780M)

Este documento detalla la viabilidad y el impacto de mover el entrenamiento de TinyThinker a la infraestructura de **Modal.com**.

## 1. El Potencial de Modal
Modal ofrece una infraestructura serverless para IA con un crédito mensual recurrente de **30$**. Para un modelo de 50M-100M parámetros, esto representa una aceleración masiva.

### Tarifas de GPU (Estimación)
| GPU | Coste/hora | Tiempo Gratis (30$) | Recomendación |
| :--- | :---: | :---: | :--- |
| **Nvidia L4 (24GB)** | $0.80 | **37.5 horas** | ★★★★☆ (Eficiencia energética) |
| **Nvidia A10G (24GB)** | $1.10 | **27.2 horas** | ★★★★★ (Balance ideal) |
| **Nvidia A100 (40GB)** | $2.10 | **14.2 horas** | ★★★☆☆ (Overkill para 50M) |
| **Nvidia H100 (80GB)** | $3.95 | **7.6 horas** | ★★☆☆☆ (Solo para batches masivos) |

## 2. Comparativa: CPU Local vs GPU Modal
Basado en datos de `NEXT_TRAINING_PLAN.md`:

| Métrica | Local (8 threads CPU) | Modal (A10G GPU) | Mejora |
| :--- | :--- | :--- | :--- |
| **10 Iteraciones** | 410 segundos (~7 min) | < 2 segundos | ~200x más rápido |
| **Epoch Completo** | Días de CPU | Minutos / 1 hora | Ahorro radical |
| **Coste** | Electricidad + Desgaste | 0$ (dentro del crédito) | Alta viabilidad |

## 3. Estrategia de Implementación
Para migrar el entrenamiento a Modal sin modificar drásticamente el flujo local:

1.  **Modal Volumes:** Subir los archivos `.bin` (TinyStories, Wikipedia) a un volumen persistente en la nube.
2.  **Decorador `@app.function`:** Usar el SDK de Modal para envolver la función de entrenamiento principal.
3.  **Checkpoints Remotos:** Configurar el script para guardar los archivos `.pt` directamente en el volumen de Modal y descargarlos solo cuando sea necesario evaluarlos localmente en la Radeon 780M.

## 4. Próximos Experimentos
*   **Prueba de Benchmarking:** Ejecutar una mini-evaluación en una L4 para confirmar la velocidad de paso de gradiente.
*   **Implementación del MoE:** Si el entrenamiento en GPU es estable, podemos explorar el entrenamiento de los "Expertos Vírgenes" en paralelo usando múltiples contenedores Modal.

---
> [!NOTE]
> *Modal factura por segundo real de ejecución. No hay pago por "instancia encendida", solo por cómputo procesado. Es ideal para iteraciones rápidas de entrenamiento.*
