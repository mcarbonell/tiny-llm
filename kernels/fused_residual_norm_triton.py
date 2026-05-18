import torch
import triton
import triton.language as tl

@triton.jit
def _fused_residual_norm_forward_kernel(
    x_ptr, y_ptr, alpha_ptr, out_ptr, norms_ptr,
    stride_xb, stride_xn,
    stride_yb, stride_yn,
    stride_ob, stride_on,
    dim,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # pointers for this row
    row_x = x_ptr + pid * stride_xb
    row_y = y_ptr + pid * stride_yb
    row_out = out_ptr + pid * stride_ob
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < dim
    
    # Load row values
    val_x = tl.load(row_x + offsets * stride_xn, mask=mask, other=0.0)
    val_y = tl.load(row_y + offsets * stride_yn, mask=mask, other=0.0)
    val_alpha = tl.load(alpha_ptr + offsets, mask=mask, other=0.0)
    
    # Compute u = x + abs(alpha) * y
    abs_alpha = tl.abs(val_alpha)
    val_u = val_x + abs_alpha * val_y
    
    # Compute sum of squares
    sum_sq = tl.sum(val_u * val_u, axis=0)
    norm = tl.sqrt(sum_sq) + 1e-8
    
    # Normalize
    val_z = val_u / norm
    
    # Store results
    tl.store(row_out + offsets * stride_on, val_z, mask=mask)
    tl.store(norms_ptr + pid, norm)

@triton.jit
def _fused_residual_norm_backward_kernel(
    grad_out_ptr, out_ptr, y_ptr, alpha_ptr, norms_ptr,
    grad_x_ptr, grad_y_ptr, grad_alpha_accum_ptr,
    stride_gb, stride_gn,
    stride_ob, stride_on,
    stride_yb, stride_yn,
    stride_gxb, stride_gxn,
    stride_gyb, stride_gyn,
    batch_seq, dim,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0) # parallelized over rows (batch * seq)
    
    row_grad_out = grad_out_ptr + pid * stride_gb
    row_out = out_ptr + pid * stride_ob
    row_y = y_ptr + pid * stride_yb
    row_grad_x = grad_x_ptr + pid * stride_gxb
    row_grad_y = grad_y_ptr + pid * stride_gyb
    
    norm = tl.load(norms_ptr + pid)
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < dim
    
    # Load values
    g_out = tl.load(row_grad_out + offsets * stride_gn, mask=mask, other=0.0)
    z = tl.load(row_out + offsets * stride_on, mask=mask, other=0.0)
    y = tl.load(row_y + offsets * stride_yn, mask=mask, other=0.0)
    alpha = tl.load(alpha_ptr + offsets, mask=mask, other=0.0)
    
    # Compute dot product s = sum(g_out * z)
    s = tl.sum(g_out * z, axis=0)
    
    # Compute grad_u = (g_out - z * s) / norm
    grad_u = (g_out - z * s) / norm
    
    # Compute grad_x = grad_u
    tl.store(row_grad_x + offsets * stride_gxn, grad_u, mask=mask)
    
    # Compute grad_y = grad_u * abs(alpha)
    abs_alpha = tl.abs(alpha)
    grad_y = grad_u * abs_alpha
    tl.store(row_grad_y + offsets * stride_gyn, grad_y, mask=mask)
    
    # Compute grad_alpha part = grad_u * y * sign(alpha)
    sign_alpha = tl.where(alpha > 0.0, 1.0, tl.where(alpha < 0.0, -1.0, 0.0))
    grad_alpha_part = grad_u * y * sign_alpha
    
    # Write grad_alpha_part to accumulator using atomic add
    tl.atomic_add(grad_alpha_accum_ptr + offsets, grad_alpha_part, mask=mask)

class FusedResidualNormTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, alpha):
        orig_shape = x.shape
        dim = orig_shape[-1]
        
        x_flat = x.view(-1, dim).contiguous()
        y_flat = y.view(-1, dim).contiguous()
        alpha_flat = alpha.contiguous()
        
        batch_seq = x_flat.size(0)
        out_flat = torch.empty_like(x_flat)
        norms = torch.empty(batch_seq, dtype=x.dtype, device=x.device)
        
        # Determine block size (power of 2 greater than or equal to dim)
        BLOCK_SIZE = triton.next_power_of_2(dim)
        
        grid = (batch_seq,)
        
        _fused_residual_norm_forward_kernel[grid](
            x_flat, y_flat, alpha_flat, out_flat, norms,
            x_flat.stride(0), x_flat.stride(1),
            y_flat.stride(0), y_flat.stride(1),
            out_flat.stride(0), out_flat.stride(1),
            dim,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        ctx.save_for_backward(out_flat, y_flat, alpha_flat, norms)
        ctx.batch_seq = batch_seq
        ctx.dim = dim
        ctx.orig_shape = orig_shape
        ctx.BLOCK_SIZE = BLOCK_SIZE
        
        return out_flat.view(orig_shape)

    @staticmethod
    def backward(ctx, grad_out):
        out_flat, y_flat, alpha_flat, norms = ctx.saved_tensors
        batch_seq = ctx.batch_seq
        dim = ctx.dim
        orig_shape = ctx.orig_shape
        BLOCK_SIZE = ctx.BLOCK_SIZE
        
        grad_out_flat = grad_out.view(-1, dim).contiguous()
        
        grad_x_flat = torch.empty_like(out_flat)
        grad_y_flat = torch.empty_like(out_flat)
        # Initialize gradient of alpha with zeros
        grad_alpha = torch.zeros(dim, dtype=out_flat.dtype, device=out_flat.device)
        
        grid = (batch_seq,)
        
        _fused_residual_norm_backward_kernel[grid](
            grad_out_flat, out_flat, y_flat, alpha_flat, norms,
            grad_x_flat, grad_y_flat, grad_alpha,
            grad_out_flat.stride(0), grad_out_flat.stride(1),
            out_flat.stride(0), out_flat.stride(1),
            y_flat.stride(0), y_flat.stride(1),
            grad_x_flat.stride(0), grad_x_flat.stride(1),
            grad_y_flat.stride(0), grad_y_flat.stride(1),
            batch_seq, dim,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return grad_x_flat.view(orig_shape), grad_y_flat.view(orig_shape), grad_alpha

def fused_residual_norm_triton(x, y, alpha):
    """
    Triton wrapper that applies FusedResidualNormTriton autograd Function.
    """
    return FusedResidualNormTriton.apply(x, y, alpha)
