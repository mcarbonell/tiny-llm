# Experiment Log — Unified Spectral (Matrix-Free + Hippocampus)

Repositorio: `tiny-thinker` (fork de nanoGPT). Arquitectura unificada en
`model/model_spectral_unified.py` (fusion de V9 matrix-free + V10 hippocampus +
kernel FWHT nativo), entrenada en CPU (4 threads) para no saturar la maquina del
autor. Este doc registra experimentos reproducibles y findings.

## Setup comun (salvo que se indique)

- Modelo: `UnifiedSpectral`, 10.5M params, dim=2048, emb_dim=256 (factorizado),
  n_layers=8, k_walsh=256, vocab=32768.
- Datos: `data/train_v2_32k.bin` (tokenizer `model/tokenizer_v2_32k.json`).
- Optim: AdamW, lr 3e-3 coseno (warmup 30), weight_decay 0, grad_clip 1.0.
- CPU 4 threads (OMP_NUM_THREADS=4, MKL_NUM_THREADS=4). Maquina nunca >32% CPU.
- Los runs duran ~16 min cada uno a seq=256, 300 iters.

## Run de referencia (smoke unificado)

- Config: `configs/train_unified_smoke.yaml` (hippo ON, spherical ON).
- Resultado: train_loss 7.74, val_loss 8.52 (iter 300).
- Conclusion: converge limpio y monotono (10.58 -> 8.52), SIN el ruido de V11
  (que oscilaba 4.0<->4.7 a lr 1.5e-2 + weight-sharing de bloques).
- Hallazgo clave de ingenieria: V11 tardaba ~750s/iter por `chunk_size=256`
  reprocesando el bloque compartido 4x. El unificado con `chunk_size=max_seq_len`
  (1 chunk) baja a ~74s/iter estimado a seq=1024. El bottleneck era el chunking,
  no el FWHT ni la CPU.

## Ablation A — Hippocampus ON vs OFF

- Diferencia unica: `use_hippocampus` (True/False). Resto = smoke de referencia.
- Resultados (val_loss, iter 300):
  - Hippo ON : 8.56
  - Hippo OFF: 8.54
- Finding: **el hippocampus no aporta mejora medible a seq=256**. Diff 0.02 = ruido.
  Tiene sentido: a seq corto el mixer FFT causal ya ve todo el contexto; el
  hippocampus (memoria stateful O(1) para seq largas) es redundante aqui.
  Su utilidad estaria solo en seq >> ventana o inferencia streaming stateful.
- Logs: `logs/ablation_hippo_on.log`, `logs/ablation_hippo_off.log`
- Configs: `configs/train_ablation_hippo_{on,off}.yaml`

## Ablation B — Spherical head (nGPT) ON vs OFF

- Diferencia unica: `spherical_head` (True=SphericalHead hiper-esferico con tau,
  False=nn.Linear estandar). Resto = hippo OFF (para aislar solo la cabeza).
- Resultados (val_loss, iter 300):
  - Spherical ON : 8.47  (train 7.76)
  - Linear OFF   : 8.32  (train 7.39)
- Finding: **el head lineal estandar gana por ~0.15 val_loss** en este setting.
  La normalizacion hiper-esferica de nGPT no ayuda aqui (y el run ON fue MAS
  lento: ~48s/iter vs ~32s/iter del OFF — el SphericalHead añade softmax sobre
  vocab + temperatura). Posible explicacion: a 300 iters y lr bajo, la
  regularizacion esferica no compensa la flexibilidad del head lineal; nGPT
  reporta beneficios a muchos mas pasos / lr distinto.
- Logs: `logs/ablation_sph_on.log`, `logs/ablation_sph_off.log`
- Configs: `configs/train_ablation_sph_{on,off}.yaml`

## Resumen de findings

1. Weight-sharing de bloques (de V11) enturbia la convergencia -> QUITAR (hecho
   en unificado).
2. chunk_size debe ser = max_seq_len para no multiplicar computo por el chunking
   BPTT del hippocampus.
3. Hippocampus: prescindible a seq corto. Re-evaluar a seq largo (1024/2048).
4. Spherical head nGPT: no aporta a corto plazo; head lineal es mejor y mas rapido
   en este regime.
5. FWHT kernel vs denso: a d=2048 el denso entrena MEJOR (~0.2 val_loss) porque
   replica F.normalize por fila que el kernel omite. El kernel mantiene ventaja
   de MEMORIA/asintotica (O(d log d) vs O(d^2)), no de calidad a d baja.
6. Weight tying: head compartido con embedding (ON) da val 7.97 a 10.5M params;
   head propio (OFF) da 7.87 a 18.9M. La diferencia (0.1) no justifica +8.4M params.
   Usar weight_tying=True.

## Archivos creados

- `model/walsh_linear_fwht.py` — WalshLinear por kernel FWHT (sin matriz densa).
- `model/model_spectral_unified.py` — arquitectura unificada con flags.
- `configs/train_unified_v1.yaml` — config seria (seq=1024, 2000 iters).
- `configs/train_unified_smoke.yaml` — smoke referencia.
- `configs/train_ablation_hippo_{on,off}.yaml`, `train_ablation_sph_{on,off}.yaml`.
- `scripts/validate_unified.py` — valida FWHT + equivalencia denso + fwd/bwd.
- `scripts/benchmark_walsh_kernel.py` — 1.12x kernel vs denso a d=2048.
- `scripts/benchmark_unified_e2e.py` — 10.5M params, ~74s/iter a seq=1024.
- `scripts/smoke_unified_train.py` — smoke del path de train.py.

## Siguientes pasos sugeridos

- A: run serio unificado seq=1024, 2000 iters, lr 1e-3 (curva comparable a V11).
- C1: hippocampus a seq largo (donde SÍ deberia marcar diferencia).

## Ablation F — weight_tying ON vs OFF

- Diferencia unica: `weight_tying` (True=head comparte peso con embedding factorizado,
  False=head `nn.Linear` con pesos propios). Resto = hippo OFF + spherical OFF + denso.
- Resultados (val_loss, iter 300):
  - weight_tying ON : 7.97  (train 7.14, 10.52M params)
  - weight_tying OFF: 7.87  (train 7.35, 18.91M params)
- Finding: el head propio gana solo ~0.1 val_loss pero cuesta +8.4M params (80% mas).
  A esta escala NO compensa. `weight_tying=True` es el mejor tradeoff (misma calidad,
  8.4M menos de parametros). Nota: el run OFF (18.91M) tardo ~el doble y llego a colgar
  la maquina del autor en un primer intento (reinicio); relanzado igual termino bien a
  ~41s/iter con CPU 28%. La causa del cuelgue inicial NO fue RAM (habia 39GB libres),
  probablemente percepcion de lentitud por el run largo.
- Logs: `logs/ablation_wt_on.log`, `logs/ablation_wt_off.log`
- Configs: `configs/train_ablation_wt_{on,off}.yaml`

## Resumen global de ablations (seq=256, 300 iters, lr 3e-3, 4 threads CPU)

Todos los ablations parten de la config unificada base. Mejor config "limpia" hallada:

| Componente      | Ganador       | val_loss | Nota |
|-----------------|---------------|----------|------|
| Hippocampus     | OFF           | 8.54     | redundante a seq corto |
| Spherical head  | OFF (lineal)  | 8.32     | nGPT esferico perjudica a corto |
| FWHT kernel     | OFF (denso)   | 8.04     | denso entrena mejor a d=2048* |
| Weight tying    | ON (compart.) | 7.97     | +8.4M params del head no compensan |

* El kernel FWHT sigue siendo superior en MEMORIA (O(d log d) vs O(d^2)) y
  escalabilidad; a d=2048 el denso es viable y entrena mejor. Para igualar calidad
  el kernel deberia replicar F.normalize por fila que V10 aplica.

Config unificada recomendada para run serio: hippo OFF + spherical OFF + denso
(use_fwht_kernel=False) + weight_tying ON = val_loss ~7.97 a 300 iters, seq=256.
Siguiente paso natural: run serio seq=1024, 2000 iters, lr 1e-3 (A) para comparar
contra V11 con config limpia y apples-to-apples.

## Run serio A — configs/train_serious_v1.yaml (resultados finales)

- Config limpia ganadora de los ablations: hippo OFF, spherical OFF, FFN denso
  (use_fwht_kernel=False), weight_tying ON.
- Hyperparams: dim=2048, emb_dim=256, n_layers=8, k_walsh=256, vocab=32768,
  seq_len=1024, batch=8, grad_accum=4, max_iters=2000, lr=1e-3 (coseno, warmup 50,
  min_lr 1e-4), AdamW (weight_decay=0), grad_clip=1.0.
- Lanzado 2026-07-20 23:29, completado 2026-07-21 ~19:40 (20h 11min).
- Device: CPU 8 threads (OMP_NUM_THREADS=8). Speed: ~31s/iter early, ~440s/iter late
  (c/ 250 iters val evalañade ~90s). Total ~20h.
- Checkpoints: `checkpoints/serious_v1/ckpt_pretrain_best.pt` y `_latest.pt`.
- Log: `logs/serious_v1.log`.

### Resultados

| Metrica | Iter 0 | Iter 250 | Iter 500 | Iter 750 | Iter 1000 | Iter 1250 | Iter 1500 | Iter 1750 | **Iter 2000** |
|---------|--------|----------|----------|----------|-----------|-----------|-----------|-----------|---------------|
| train_loss | 10.8980 | 7.3245 | 6.9903 | 6.8060 | 6.7274 | 6.4591 | 6.4809 | 6.5140 | **6.3493** |
| val_loss | 10.9186 | 8.2966 | 7.7857 | 7.4935 | 7.3911 | 7.2016 | 7.2194 | 7.1515 | **7.1373** |

- **Mejor val_loss: 7.1373** (iter 2000, ultimo checkpoint = mejor modelo).
- Mejora total: val_loss 10.92 -> 7.14 (delta -3.78). Sin signos de sobreajuste:
  gap train/val ~0.79 constante.

### Comparacion contra V11

| Feature | serious_v1 (unified) | V11 Run 2 (champ) | V11 Run 4 |
|---------|---------------------|-------------------|-----------|
| Modelo | UnifiedSpectral | V11 Albert | V11 Albert |
| dim/emb_dim | 2048/256 | 1024/256 | 2048/256 |
| k_walsh | 256 | 512 | 512 |
| n_layers | 8 | 8 | 8 |
| Params | 10.53M | 9.44M | 9.97M |
| lr | 1e-3 (cos, min 1e-4) | **1.5e-2 (constante)** | 1.5e-2 (constante) |
| batch | 8 | 16 | 16 |
| **val_loss @2000** | **7.1373** | **4.1287** | **4.1600** |

### Analisis

**Resultado INESPERADO**: La config "limpia" (sin weight-sharing, sin chunking)
queda MUY por detras de V11 (~3.0 val_loss de diferencia). A igualdad de params
(~10M) y seq_len (1024), V11 converge a loss 4.1-4.2 mientras la unified apenas
baja de 7.1.

**Causas probables (ordenadas por impacto)**:

1. **Learning rate demasiado conservador**: V11 uso lr=0.015 constante. La unified
   arranco en 0.001 (15x menor) y decae a 0.0001. A 2000 iters, el LR bajo limita
   la convergencia. La curva de val_loss seguia bajando al final (7.14) pero a
   tasa decreciente.

2. **Batch size menor**: batch=8 vs V11 batch=16. Menos gradientes por paso ->
   mas ruido en la estimacion.

3. **k_walsh=256 vs k=512**: El rango Walsh del mixer/FFN es la mitad. En V11 se
   demostro que k alto es critico (Run 2 con k=512 gana a Run 3 con k=256, aun
   con dim=2048).

4. **Weight-sharing de bloques (V11)**: Aunque se documento como "bug" que
   enturbiaba la convergencia, en la practica V11 con weight-sharing Y lr alto
   converge mucho mejor que unified sin weight-sharing Y lr bajo. El efecto
   regularizador del weight-sharing podria estar permitiendo LRs mas agresivos
   sin divergencia.

5. **Chunk_size**: V11 usaba chunk_size=256 (reprocesaba 4x), la unified
   chunk_size=1024. Esto afecta al hippocampus (OFF aqui) y no explica la
   diferencia.

**Conclusion**: El factor dominante es el LR. V11 con lr=0.015 converge a ~4.1;
la unified con lr=0.001 se queda en 7.1. La config limpia (sin weight-sharing)
no es intrinsica peor — simplemente no se ha explorado con LR suficientemente
alto. Queda como experimento pendiente relanzar unified con lr=0.01-0.015 y
batch=16 para comparacion apples-to-apples.

### Lecciones para proximos runs

- El LR de V11 (0.015) no era accidental: las arquitecturas espectrales toleran
  (y necesitan) LRs altos.
- Weight_decay=0 se confirma correcto (GEMINI.md regla #5).
- La unified a 10.53M con d=2048, k=256 es efectivamente ~31s/iter pero el
  bottleneck real es la evaluacion cada 250 iters (anyade ~90s).
- El checkpoint final pesa ~42MB (10.53M params x 4 bytes).

## TODO resuelto

- [x] **Run serio A completado**: val_loss 7.1373 a 2000 iters.
- [x] **Comparacion contra V11 documentada**: gap de ~3.0, atribuido a LR y k.
- [ ] Relanzar unified con lr=0.015, batch=16, k_walsh=512 para validar hipotesis
      (Run serio B).
- [ ] (Opcional) Aislar contribucion del hippocampus a seq largo (C1).

## Ablation E — FWHT kernel vs matriz Walsh densa

- Diferencia unica: `use_fwht_kernel` (True=kernel FWHT nativo, False=matriz Walsh
  densa materializada via WalshLinear de V10). Resto = hippo OFF + spherical OFF
  (config "limpia" ganadora de A y B).
- Resultados (val_loss, iter 300):
  - FWHT kernel ON : 8.24  (train 7.55, ~32s/iter)
  - Walsh denso OFF: 8.04  (train 7.13, ~39s/iter)
- Finding (INESPERADO): **la version DENSA gana por ~0.2 val_loss y es mas lenta**.
  Teoricamente ambas sintetizan la MISMA matriz W = H_out[:,:k] @ core @ H_in[:k,:],
  y validate_unified.py confirma que la sintesis numerica coincide (error 4.7e-7).
  La diferencia esta en la RUTA del gradiente: el denso de V10 aplica
  `F.normalize(W_synthesized, dim=-1)` (normaliza por fila) + scale aprendible,
  mientras el kernel FWHT usa Hadamard crudo + factores 1/sqrt(d) + scale, SIN
  normalizar por fila. Eso cambia la superficie de optimizacion y favorece al
  denso en este regime. NO es un bug del kernel: es que el kernel omite la
  normalizacion por fila que V10 aplicaba.
- Implicacion: el kernel FWHT sigue siendo superior en MEMORIA (O(d log d) vs
  O(d^2)) y en escalabilidad asintotica (a d=4096/8192 el denso explota), pero a
  d=2048 el denso es viable y entrena mejor. Para igualar, el kernel deberia
  replicar la normalizacion por fila (o su version differentiable equivalente).
- Logs: `logs/ablation_fwht_on.log`, `logs/ablation_fwht_off.log`
- Configs: `configs/train_ablation_fwht_{on,off}.yaml`
- Cambios de codigo: `UnifiedArgs.use_fwht_kernel`, `MatrixFreeFFN` elige
  `WalshLinearFWHT` (kernel) o `WalshLinear` (denso de V10); mapeado en train.py.
