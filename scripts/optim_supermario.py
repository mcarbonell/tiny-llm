import math
import torch
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F

def create_dct_matrix(N, dtype, device):
    """Crea una matriz DCT-II de tamaño N x N."""
    n = torch.arange(N, dtype=dtype, device=device)
    k = torch.arange(N, dtype=dtype, device=device).unsqueeze(1)
    dct_mat = torch.cos(math.pi / N * (n + 0.5) * k)
    dct_mat[0] *= 1.0 / math.sqrt(2.0)
    dct_mat *= math.sqrt(2.0 / N)
    return dct_mat

def dct_2d(x, dct_mat_h, dct_mat_w):
    """Proyección al dominio espectral DCT."""
    return torch.matmul(dct_mat_h, torch.matmul(x, dct_mat_w.t()))

def idct_2d(x_dct, dct_mat_h, dct_mat_w):
    """Proyección inversa al dominio espacial."""
    return torch.matmul(dct_mat_h.t(), torch.matmul(x_dct, dct_mat_w))

class SuperMarioOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, k_ratio=0.25):
        """
        k_ratio: Fracción de la resolución original a mantener.
        """
        if not 0.0 < k_ratio <= 1.0:
            raise ValueError(f"Invalid k_ratio: {k_ratio}")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, k_ratio=k_ratio)
        super(SuperMarioOptimizer, self).__init__(params, defaults)
        self._dct_cache = {}

    def _get_dct_matrices(self, h, w, dtype, device):
        key = (h, w, dtype, device)
        if key not in self._dct_cache:
            self._dct_cache[key] = (create_dct_matrix(h, dtype, device), create_dct_matrix(w, dtype, device))
        return self._dct_cache[key]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                
                grad = p.grad
                state = self.state[p]
                k_ratio = group['k_ratio']

                # Inicialización del estado
                if len(state) == 0:
                    state['step'] = 0
                    # Detectar si es un parámetro espectral 'core' (V7/V6) o una matriz densa
                    is_core = any(name in str(p) for name in ['core']) # Heurística basada en nombrado
                    
                    if grad.dim() == 2 and grad.shape[0] >= 32 and grad.shape[1] >= 32:
                        state['is_compressed'] = True
                        state['is_spectral'] = is_core
                        state['orig_shape'] = grad.shape
                        
                        comp_h = max(1, int(grad.shape[0] * k_ratio))
                        comp_w = max(1, int(grad.shape[1] * k_ratio))
                        state['comp_shape'] = (comp_h, comp_w)
                        
                        state['exp_avg'] = torch.zeros((comp_h, comp_w), dtype=grad.dtype, device=grad.device)
                        state['exp_avg_sq'] = torch.zeros((comp_h, comp_w), dtype=grad.dtype, device=grad.device)
                    else:
                        state['is_compressed'] = False
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                if state['is_compressed']:
                    orig_h, orig_w = state['orig_shape']
                    comp_h, comp_w = state['comp_shape']
                    
                    if state.get('is_spectral', False):
                        # MODO ESPECTRAL: Compresión directa en frecuencia (DCT)
                        # No usamos adaptive_avg_pool porque destruiría los coeficientes.
                        # En su lugar, proyectamos el gradiente al espacio reducido de DCT.
                        mat_h, mat_w = self._get_dct_matrices(comp_h, comp_w, grad.dtype, grad.device)
                        
                        # El gradiente de un 'core' ya está en dominio de frecuencia,
                        # pero al ser Matrix-Free, lo tratamos como espacial para SMO
                        # y extraemos sus componentes de baja frecuencia.
                        g_comp = F.adaptive_avg_pool2d(grad.unsqueeze(0).unsqueeze(0), (comp_h, comp_w)).squeeze(0).squeeze(0)
                        g_dct = dct_2d(g_comp, mat_h, mat_w)
                        
                        exp_avg.mul_(beta1).add_(g_dct, alpha=1 - beta1)
                        
                        g_sq_comp = F.adaptive_avg_pool2d((grad**2).unsqueeze(0).unsqueeze(0), (comp_h, comp_w)).squeeze(0).squeeze(0)
                        g_sq_dct = dct_2d(g_sq_comp, mat_h, mat_w)
                        exp_avg_sq.mul_(beta2).add_(g_sq_dct, alpha=1 - beta2)
                        
                        m_rec = idct_2d(exp_avg, mat_h, mat_w)
                        v_rec = idct_2d(exp_avg_sq, mat_h, mat_w)
                    else:
                        # MODO ESPACIAL: Bilinear (Básico para matrices densas)
                        g_view = grad.unsqueeze(0).unsqueeze(0)
                        g_comp = F.adaptive_avg_pool2d(g_view, (comp_h, comp_w)).squeeze(0).squeeze(0)
                        exp_avg.mul_(beta1).add_(g_comp, alpha=1 - beta1)
                        
                        g_sq_comp = F.adaptive_avg_pool2d((grad**2).unsqueeze(0).unsqueeze(0), (comp_h, comp_w)).squeeze(0).squeeze(0)
                        exp_avg_sq.mul_(beta2).add_(g_sq_comp, alpha=1 - beta2)
                        
                        m_rec = exp_avg
                        v_rec = exp_avg_sq

                    # Reconstrucción al tamaño original
                    m_view = m_rec.unsqueeze(0).unsqueeze(0)
                    v_view = v_rec.unsqueeze(0).unsqueeze(0)
                    m_rec = F.interpolate(m_view, size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                    v_rec = F.interpolate(v_view, size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                    v_rec = torch.clamp(v_rec, min=0.0)

                else:
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    m_rec, v_rec = exp_avg, exp_avg_sq

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bias_correction1
                denom = (v_rec.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                p.addcdiv_(m_rec, denom, value=-step_size)

        return loss
