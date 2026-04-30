"""
benchmark_spectral_vs_dense.py — EXP-6

Mide y compara rigorosamente SpectralThinker vs Dense Transformer equivalente:
  - Parámetros totales y por componente
  - Velocidad de forward (inferencia)
  - Velocidad de training step (forward + backward + optimizer)
  - Tiempo del optimizer step aislado
  - Memoria de pesos y estado del optimizer

Ejecutar:
    python scripts/benchmark_spectral_vs_dense.py
"""

import sys, time, math, gc
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')
from model.model_spectral import SpectralThinker, SpectralArgs

# ─── Configuración del benchmark ─────────────────────────────────────────────
DIM         = 256
N_LAYERS    = 6
N_HEADS     = 8
N_KV_HEADS  = 4
VOCAB_SIZE  = 16384
K           = 64
BATCH       = 16
SEQ_LEN     = 256
N_WARMUP    = 3
N_MEASURE   = 10
DEVICE      = torch.device('cpu')


# ─── Modelo Denso equivalente ─────────────────────────────────────────────────

class DenseAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep      = n_heads // n_kv_heads
        self.head_dim   = dim // n_heads
        self.wq = nn.Linear(dim, n_heads    * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim,    bias=False)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.wq(x).view(B, S, self.n_heads,    self.head_dim).transpose(1,2)
        k = self.wk(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1,2)
        v = self.wv(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1,2)
        if self.n_rep > 1:
            k = k[:,:,None,:,:].expand(B,self.n_kv_heads,self.n_rep,S,self.head_dim).reshape(B,self.n_heads,S,self.head_dim)
            v = v[:,:,None,:,:].expand(B,self.n_kv_heads,self.n_rep,S,self.head_dim).reshape(B,self.n_heads,S,self.head_dim)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out.transpose(1,2).reshape(B, S, -1))


class DenseFeedForward(nn.Module):
    def __init__(self, dim, multiple_of=256, ffn_dim_multiplier=2.0):
        super().__init__()
        hidden = int(2 * (4 * dim) / 3)
        hidden = int(ffn_dim_multiplier * hidden)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DenseBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        self.attn = DenseAttention(dim, n_heads, n_kv_heads)
        self.ffn  = DenseFeedForward(dim)
        self.n1   = nn.LayerNorm(dim)
        self.n2   = nn.LayerNorm(dim)

    def forward(self, x):
        h = x + self.attn(self.n1(x))
        return h + self.ffn(self.n2(h))


class DenseTransformer(nn.Module):
    def __init__(self, dim, n_layers, n_heads, n_kv_heads, vocab_size):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([DenseBlock(dim, n_heads, n_kv_heads) for _ in range(n_layers)])
        self.norm   = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        # Matrices separadas (igual que Nano v1)

    def forward(self, tokens, targets=None):
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x)
        logits = self.output(self.norm(x))
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits


# ─── Utilidades de medición ───────────────────────────────────────────────────

def param_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())

def optimizer_state_bytes(optimizer):
    """Estima el tamaño del estado de Adam (m + v por parámetro)."""
    total = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            total += 2 * p.numel() * p.element_size()  # m y v
    return total

def time_forward(model, tokens, n=N_MEASURE):
    """Mide tiempo de forward puro (sin gradientes)."""
    model.eval()
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = model(tokens)
    times = []
    with torch.no_grad():
        for _ in range(n):
            t0 = time.perf_counter()
            _ = model(tokens)
            times.append(time.perf_counter() - t0)
    return times

def time_train_step(model, optimizer, tokens, targets, n=N_MEASURE):
    """Mide tiempo de step completo: forward + backward + optimizer."""
    model.train()
    for _ in range(N_WARMUP):
        _, loss = model(tokens, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    times_total   = []
    times_optim   = []
    for _ in range(n):
        optimizer.zero_grad()
        t0 = time.perf_counter()
        _, loss = model(tokens, targets)
        loss.backward()
        t_before_optim = time.perf_counter()
        optimizer.step()
        t1 = time.perf_counter()
        times_total.append(t1 - t0)
        times_optim.append(t1 - t_before_optim)
    return times_total, times_optim

def stats(times):
    import statistics
    avg = sum(times) / len(times)
    std = statistics.stdev(times) if len(times) > 1 else 0
    return avg, std

def fmt_ms(t): return '{:.1f}ms'.format(t * 1000)
def fmt_mb(b): return '{:.2f}MB'.format(b / 1e6)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('EXP-6: SpectralThinker vs Dense — Benchmark')
    print('Batch={}, SeqLen={}, Dim={}, K={}, Layers={}'.format(
        BATCH, SEQ_LEN, DIM, K, N_LAYERS))
    print('Warmup={}, Mediciones={}'.format(N_WARMUP, N_MEASURE))
    print('=' * 60)

    # Datos sintéticos reproducibles
    torch.manual_seed(42)
    tokens  = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), device=DEVICE)
    targets = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), device=DEVICE)

    # ── SpectralThinker ──────────────────────────────────────────────────────
    print('\n[1/2] Construyendo SpectralThinker Nano (dim={}, k={})...'.format(DIM, K))
    spec_args = SpectralArgs(
        dim=DIM, n_layers=N_LAYERS, n_heads=N_HEADS, n_kv_heads=N_KV_HEADS,
        vocab_size=VOCAB_SIZE, max_seq_len=SEQ_LEN,
        k_dim_attn=K, k_dim_ffn=K, k_hidden_ffn=K*2
    )
    spectral = SpectralThinker(spec_args).to(DEVICE)
    spec_params   = sum(p.numel() for p in spectral.parameters())
    spec_proj     = sum(p.numel() for n, p in spectral.named_parameters() if 'core' in n)
    spec_model_mb = param_bytes(spectral)
    spec_opt      = torch.optim.AdamW(spectral.parameters(), lr=1e-3, foreach=False)
    spec_opt_mb   = optimizer_state_bytes(spec_opt)

    print('  Params totales:      {:>12,}  ({:.2f}M)'.format(spec_params, spec_params/1e6))
    print('  Nucleos espectrales: {:>12,}  ({:.1f}K)'.format(spec_proj, spec_proj/1e3))
    print('  Pesos en RAM:        {:>12}'.format(fmt_mb(spec_model_mb)))
    print('  Estado optimizer:    {:>12}'.format(fmt_mb(spec_opt_mb)))

    print('  Midiendo forward...', end=' ', flush=True)
    spec_fwd = time_forward(spectral, tokens)
    print('OK')
    print('  Midiendo train step...', end=' ', flush=True)
    spec_total, spec_opt_t = time_train_step(spectral, spec_opt, tokens, targets)
    print('OK')

    # ── Dense Transformer ────────────────────────────────────────────────────
    print('\n[2/2] Construyendo Dense Transformer (dim={}, full rank)...'.format(DIM))
    dense = DenseTransformer(DIM, N_LAYERS, N_HEADS, N_KV_HEADS, VOCAB_SIZE).to(DEVICE)
    dense_params   = sum(p.numel() for p in dense.parameters())
    dense_proj     = sum(p.numel() for n, p in dense.named_parameters()
                        if any(x in n for x in ['wq','wk','wv','wo','w1','w2','w3']))
    dense_model_mb = param_bytes(dense)
    dense_opt      = torch.optim.AdamW(dense.parameters(), lr=1e-3, foreach=False)
    dense_opt_mb   = optimizer_state_bytes(dense_opt)

    print('  Params totales:      {:>12,}  ({:.2f}M)'.format(dense_params, dense_params/1e6))
    print('  Proyecciones densas: {:>12,}  ({:.2f}M)'.format(dense_proj, dense_proj/1e6))
    print('  Pesos en RAM:        {:>12}'.format(fmt_mb(dense_model_mb)))
    print('  Estado optimizer:    {:>12}'.format(fmt_mb(dense_opt_mb)))

    print('  Midiendo forward...', end=' ', flush=True)
    dense_fwd = time_forward(dense, tokens)
    print('OK')
    print('  Midiendo train step...', end=' ', flush=True)
    dense_total, dense_opt_t = time_train_step(dense, dense_opt, tokens, targets)
    print('OK')

    # ── Tabla comparativa ────────────────────────────────────────────────────
    sf_avg, sf_std     = stats(spec_fwd)
    df_avg, df_std     = stats(dense_fwd)
    st_avg, st_std     = stats(spec_total)
    dt_avg, dt_std     = stats(dense_total)
    so_avg, so_std     = stats(spec_opt_t)
    do_avg, do_std     = stats(dense_opt_t)

    print('\n' + '=' * 60)
    print('RESULTADOS BENCHMARK')
    print('=' * 60)

    print('\n{:<30} {:>14} {:>14} {:>10}'.format('Metrica', 'SpectralNano', 'Dense', 'Ratio'))
    print('-' * 70)

    def row(label, sv_raw, dv_raw, sv_fmt, dv_fmt):
        ratio = dv_raw / sv_raw if sv_raw > 0 else float('inf')
        print('{:<32} {:>13} {:>13} {:>8.1f}x'.format(label, sv_fmt, dv_fmt, ratio))

    row('Params totales', spec_params, dense_params,
        '{:.2f}M'.format(spec_params/1e6), '{:.2f}M'.format(dense_params/1e6))
    row('Proyecciones espectrales vs dense', spec_proj, dense_proj,
        '{:.0f}K'.format(spec_proj/1e3), '{:.2f}M'.format(dense_proj/1e6))
    row('Pesos en RAM', spec_model_mb, dense_model_mb,
        fmt_mb(spec_model_mb), fmt_mb(dense_model_mb))
    row('Estado optimizer', spec_opt_mb, dense_opt_mb,
        fmt_mb(spec_opt_mb), fmt_mb(dense_opt_mb))
    print('-' * 70)
    row('Forward / inferencia', sf_avg, df_avg, fmt_ms(sf_avg), fmt_ms(df_avg))
    row('Train step (fwd+bwd+opt)', st_avg, dt_avg, fmt_ms(st_avg), fmt_ms(dt_avg))
    row('Optimizer step solo', so_avg, do_avg, fmt_ms(so_avg), fmt_ms(do_avg))

    print('\n' + '=' * 60)
    print('COMPRESION DE PROYECCIONES: {:.0f}x'.format(dense_proj / spec_proj))
    print('SPEEDUP TRAIN STEP:         {:.1f}x'.format(dt_avg / st_avg))
    print('SPEEDUP OPTIMIZER SOLO:     {:.1f}x'.format(do_avg / so_avg))
    print('AHORRO MEMORIA OPTIMIZER:   {:.1f}x'.format(dense_opt_mb / spec_opt_mb))
    print('=' * 60)

    # Guardar resultados como JSON
    import json, os, datetime
    results = {
        'timestamp': datetime.datetime.now().isoformat(),
        'config': {'dim': DIM, 'k': K, 'n_layers': N_LAYERS, 'batch': BATCH, 'seq_len': SEQ_LEN},
        'spectral': {
            'total_params': spec_params,
            'spectral_proj_params': spec_proj,
            'model_mb': round(spec_model_mb/1e6, 3),
            'optimizer_state_mb': round(spec_opt_mb/1e6, 3),
            'forward_ms_avg': round(sf_avg*1000, 2),
            'forward_ms_std': round(sf_std*1000, 2),
            'train_step_ms_avg': round(st_avg*1000, 2),
            'train_step_ms_std': round(st_std*1000, 2),
            'optim_step_ms_avg': round(so_avg*1000, 2),
        },
        'dense': {
            'total_params': dense_params,
            'proj_params': dense_proj,
            'model_mb': round(dense_model_mb/1e6, 3),
            'optimizer_state_mb': round(dense_opt_mb/1e6, 3),
            'forward_ms_avg': round(df_avg*1000, 2),
            'forward_ms_std': round(df_std*1000, 2),
            'train_step_ms_avg': round(dt_avg*1000, 2),
            'train_step_ms_std': round(dt_std*1000, 2),
            'optim_step_ms_avg': round(do_avg*1000, 2),
        },
        'ratios': {
            'proj_compression': round(dense_proj / spec_proj, 1),
            'train_step_speedup': round(dt_avg / st_avg, 2),
            'optimizer_speedup': round(do_avg / so_avg, 2),
            'optimizer_memory_saving': round(dense_opt_mb / spec_opt_mb, 2),
        }
    }
    os.makedirs('results/raw', exist_ok=True)
    path = 'results/raw/exp6_benchmark_spectral_vs_dense.json'
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print('\nResultados guardados en:', path)


if __name__ == '__main__':
    main()
