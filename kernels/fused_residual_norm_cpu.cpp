#include <cmath>
#include <omp.h>
#include <iostream>

/**
 * Fused Residual Update + Spherical Normalization - C++ Native Implementation
 * Compilable with: g++ -O3 -shared -fopenmp -mavx2 fused_residual_norm_cpu.cpp -o fused_residual_norm_cpu.dll
 */

extern "C" {
    // Forward Pass
    void fused_residual_norm_forward_cpu(
        const float* x,      // [B*T, D]
        const float* y,      // [B*T, D]
        const float* alpha,  // [D]
        float* out,          // [B*T, D]
        float* norms,        // [B*T]
        int batch_seq,       // B * T
        int dim              // D
    ) {
        #pragma omp parallel for
        for (int i = 0; i < batch_seq; i++) {
            const float* row_x = x + (size_t)i * dim;
            const float* row_y = y + (size_t)i * dim;
            float* row_out = out + (size_t)i * dim;
            
            float sum_sq = 0.0f;
            for (int j = 0; j < dim; j++) {
                float a = std::abs(alpha[j]);
                float val = row_x[j] + a * row_y[j];
                row_out[j] = val;
                sum_sq += val * val;
            }
            
            float norm = std::sqrt(sum_sq) + 1e-8f;
            norms[i] = norm;
            
            float inv_norm = 1.0f / norm;
            for (int j = 0; j < dim; j++) {
                row_out[j] *= inv_norm;
            }
        }
    }

    // Backward Pass
    void fused_residual_norm_backward_cpu(
        const float* grad_out, // [B*T, D]
        const float* out,      // [B*T, D] (normalized z)
        const float* y,        // [B*T, D]
        const float* alpha,    // [D]
        const float* norms,    // [B*T]
        float* grad_x,         // [B*T, D]
        float* grad_y,         // [B*T, D]
        float* grad_alpha,     // [D] (to be accumulated)
        int batch_seq,
        int dim
    ) {
        // Zero the grad_alpha buffer
        for (int j = 0; j < dim; j++) {
            grad_alpha[j] = 0.0f;
        }

        int num_threads = omp_get_max_threads();
        // Allocate a thread-local accumulation matrix to avoid atomic write collisions
        float* local_grad_alpha = new float[(size_t)num_threads * dim]();

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            float* thread_grad_alpha = local_grad_alpha + (size_t)tid * dim;

            #pragma omp for
            for (int i = 0; i < batch_seq; i++) {
                const float* row_grad_out = grad_out + (size_t)i * dim;
                const float* row_out = out + (size_t)i * dim;
                const float* row_y = y + (size_t)i * dim;
                float* row_grad_x = grad_x + (size_t)i * dim;
                float* row_grad_y = grad_y + (size_t)i * dim;
                float norm = norms[i];
                float inv_norm = 1.0f / norm;

                // 1. Compute dot product s = sum(grad_out * out)
                float s = 0.0f;
                for (int j = 0; j < dim; j++) {
                    s += row_grad_out[j] * row_out[j];
                }

                // 2. Compute grad_u = (grad_out - out * s) / norm
                //    Compute grad_x = grad_u
                //    Compute grad_y = grad_u * abs(alpha)
                //    Accumulate grad_alpha = grad_u * y * sign(alpha)
                for (int j = 0; j < dim; j++) {
                    float gu = (row_grad_out[j] - row_out[j] * s) * inv_norm;
                    row_grad_x[j] = gu;
                    
                    float a = alpha[j];
                    float abs_a = std::abs(a);
                    row_grad_y[j] = gu * abs_a;

                    float sign_a = (a > 0.0f) ? 1.0f : ((a < 0.0f) ? -1.0f : 0.0f);
                    thread_grad_alpha[j] += gu * row_y[j] * sign_a;
                }
            }
        }

        // Reduce thread-local accumulations to main grad_alpha
        for (int t = 0; t < num_threads; t++) {
            const float* thread_grad_alpha = local_grad_alpha + (size_t)t * dim;
            for (int j = 0; j < dim; j++) {
                grad_alpha[j] += thread_grad_alpha[j];
            }
        }

        delete[] local_grad_alpha;
    }
}
