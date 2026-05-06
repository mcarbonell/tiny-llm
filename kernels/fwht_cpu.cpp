#include <cmath>
#include <omp.h>

/**
 * Fast Walsh-Hadamard Transform (FWHT) - Native C++ Implementation
 * Compilable con g++ -O3 -shared -fopenmp
 */

extern "C" {
    // Interfaz C pura para ser llamada vía ctypes
    void fwht_float(float* data, int batch_size, int n) {
        float scale = 1.0f / std::sqrt((float)n);
        
        // Paralelización por batch usando OpenMP
        #pragma omp parallel for
        for (int b = 0; b < batch_size; b++) {
            float* b_data = data + (size_t)b * n;
            
            // Algoritmo de mariposa iterativo
            for (int h = 1; h < n; h <<= 1) {
                for (int i = 0; i < n; i += (h << 1)) {
                    for (int j = i; j < i + h; j++) {
                        float x_j = b_data[j];
                        float x_jh = b_data[j + h];
                        b_data[j] = x_j + x_jh;
                        b_data[j + h] = x_j - x_jh;
                    }
                }
            }
            
            // Normalización ortogonal
            for (int i = 0; i < n; i++) {
                b_data[i] *= scale;
            }
        }
    }
}
