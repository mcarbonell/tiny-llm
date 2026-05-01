"""
optim_swo.py — Smooth Walsh Optimizer (SWO) / SmoothAdam

Implementación basada en los hallazgos V125 (Smooth Spectral Adam).
Este optimizador reduce drásticamente el consumo de memoria RAM (hasta un 93%)
comprimiendo los estados históricos del gradiente (Momentum 'm' y Varianza 'v')
utilizando interpolación bilineal (proxy espacial para compresión espectral de baja frecuencia).

La "pérdida de resolución" actúa como un denoiser (regularizador implícito),
filtrando el ruido estocástico de alta frecuencia de los mini-batches.
"""

import math
import torch
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F

class SmoothAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, k_ratio=0.25):
        """
        k_ratio: Fracción de la resolución original a mantener (ej. 0.25 = 25% por dimensión).
                 Para tensores 2D, el ahorro de RAM es de 1 - (k_ratio^2) = 93.75%.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 < k_ratio <= 1.0:
            raise ValueError(f"Invalid k_ratio: {k_ratio}. Must be in (0, 1]")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, k_ratio=k_ratio)
        super(SmoothAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('SmoothAdam no soporta gradientes sparse.')

                state = self.state[p]
                k_ratio = group['k_ratio']

                # Inicialización del estado
                if len(state) == 0:
                    state['step'] = 0
                    # Solo comprimimos tensores 2D grandes (matrices de pesos).
                    # Los biases (1D) o embeddings muy asimétricos se mantienen full-res
                    # para evitar inestabilidades y porque su impacto en RAM es mínimo.
                    if grad.dim() == 2 and grad.shape[0] >= 32 and grad.shape[1] >= 32:
                        state['is_compressed'] = True
                        new_h = max(1, int(grad.shape[0] * k_ratio))
                        new_w = max(1, int(grad.shape[1] * k_ratio))
                        # Estados iniciales comprimidos
                        state['exp_avg'] = torch.zeros((new_h, new_w), dtype=grad.dtype, device=grad.device)
                        state['exp_avg_sq'] = torch.zeros((new_h, new_w), dtype=grad.dtype, device=grad.device)
                        state['orig_shape'] = grad.shape
                    else:
                        state['is_compressed'] = False
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                if state['is_compressed']:
                    # 1. Comprimir el gradiente actual (Downsample)
                    # Interpolación requiere formato (B, C, H, W)
                    g_view = grad.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                    
                    # Usamos 'area' (adaptive_avg_pool2d equivale a downsample por área)
                    # que es más estable matemáticamente para gradientes que el bilineal
                    g_comp = F.adaptive_avg_pool2d(g_view, exp_avg.shape).squeeze(0).squeeze(0)
                    
                    # 2. Actualizar momentos comprimidos
                    exp_avg.mul_(beta1).add_(g_comp, alpha=1 - beta1)
                    
                    # Para el segundo momento, comprimimos el gradiente al cuadrado
                    g_sq_comp = F.adaptive_avg_pool2d((grad ** 2).unsqueeze(0).unsqueeze(0), exp_avg_sq.shape).squeeze(0).squeeze(0)
                    exp_avg_sq.mul_(beta2).add_(g_sq_comp, alpha=1 - beta2)

                    # 3. Descomprimir (Upsample) para aplicar el update
                    # Interpolación bilineal suave para reconstruir el tensor original
                    m_view = exp_avg.unsqueeze(0).unsqueeze(0)
                    v_view = exp_avg_sq.unsqueeze(0).unsqueeze(0)
                    
                    m_rec = F.interpolate(m_view, size=state['orig_shape'], mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                    v_rec = F.interpolate(v_view, size=state['orig_shape'], mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                    
                    # Garantizar positividad estricta en v_rec (mitigar artefactos de interpolación)
                    v_rec = torch.clamp(v_rec, min=0.0)

                else:
                    # Fallback estándar para tensores 1D / pequeños
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    m_rec = exp_avg
                    v_rec = exp_avg_sq

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] / bias_correction1

                # Update final de los pesos
                denom = (v_rec.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                p.addcdiv_(m_rec, denom, value=-step_size)

        return loss
