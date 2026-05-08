
import torch
import triton
import triton.language as tl

@triton.jit
def _fwht_stage_kernel(
    x_ptr,      # Entrada
    y_ptr,      # Salida
    stride_xb, stride_xn,
    h,          # Distancia de la mariposa (constante)
    norm_factor,
    is_final: tl.constexpr,
):
    pid = tl.program_id(0)
    row_x = x_ptr + pid * stride_xb
    row_y = y_ptr + pid * stride_xb
    
    offsets = tl.arange(0, 32768)
    
    # 1. Leemos mi valor y el de mi pareja
    # offsets ^ h nos da el índice del compañero de forma perfecta
    val_self = tl.load(row_x + offsets * stride_xn)
    val_partner = tl.load(row_x + (offsets ^ h) * stride_xn)
    
    # 2. Identificamos si somos el de la izquierda (a) o derecha (b)
    mask = (offsets // h) % 2
    
    # 3. Operación de Walsh: a+b y a-b
    res = tl.where(mask == 0, val_self + val_partner, val_partner - val_self)
    
    # 4. Normalización si es la última etapa
    if is_final:
        res = res * norm_factor
        
    # 5. Guardamos en el buffer de salida (Ping-Pong)
    tl.store(row_y + offsets * stride_xn, res)

def fwht_triton(x):
    if not x.is_cuda: return x
    orig_shape = x.shape
    x_flat = x.view(-1, 32768).contiguous()
    
    # Creamos un buffer temporal para el ping-pong
    # Esto evita condiciones de carrera y asegura precisión total
    y_flat = torch.empty_like(x_flat)
    
    norm_factor = 1.0 / (32768 ** 0.5)
    grid = (x_flat.shape[0],)
    
    # Ejecutamos las 15 etapas (Ping-Pong entre x_flat y y_flat)
    current_in = x_flat
    current_out = y_flat
    
    for i in range(15):
        h = 1 << i
        is_final = (i == 14)
        
        _fwht_stage_kernel[grid](
            current_in, current_out,
            x_flat.stride(0), x_flat.stride(1),
            h, norm_factor,
            is_final=is_final
        )
        
        # Intercambiamos buffers
        current_in, current_out = current_out, current_in
        
    # El resultado final estará en current_in (porque el último swap lo movió ahí)
    return current_in.view(orig_shape)
