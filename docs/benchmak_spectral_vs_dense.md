 python scripts/benchmark_spectral_vs_dense.py
============================================================
EXP-6: SpectralThinker vs Dense — Benchmark
Batch=16, SeqLen=256, Dim=256, K=64, Layers=6
Warmup=3, Mediciones=10
============================================================

[1/2] Construyendo SpectralThinker Nano (dim=256, k=64)...
  Params totales:         4,443,392  (4.44M)
  Nucleos espectrales:      245,760  (245.8K)
  Pesos en RAM:             17.77MB
  Estado optimizer:         35.55MB
  Midiendo forward... OK
  Midiendo train step... OK

[2/2] Construyendo Dense Transformer (dim=256, full rank)...
  Params totales:        16,652,800  (16.65M)
  Proyecciones densas:    8,257,536  (8.26M)
  Pesos en RAM:             66.61MB
  Estado optimizer:        133.22MB
  Midiendo forward... OK
  Midiendo train step... OK

============================================================
RESULTADOS BENCHMARK
============================================================

Metrica                          SpectralNano          Dense      Ratio
----------------------------------------------------------------------
Params totales                           4.44M        16.65M      3.7x
Proyecciones espectrales vs dense          246K         8.26M     33.6x
Pesos en RAM                           17.77MB       66.61MB      3.7x
Estado optimizer                       35.55MB      133.22MB      3.7x
----------------------------------------------------------------------
Forward / inferencia                   330.1ms       320.4ms      1.0x
Train step (fwd+bwd+opt)              1304.5ms      1459.7ms      1.1x
Optimizer step solo                      9.6ms        25.6ms      2.7x

============================================================
COMPRESION DE PROYECCIONES: 34x
SPEEDUP TRAIN STEP:         1.1x
SPEEDUP OPTIMIZER SOLO:     2.7x
AHORRO MEMORIA OPTIMIZER:   3.7x
============================================================

Resultados guardados en: results/raw/exp6_benchmark_spectral_vs_dense.json