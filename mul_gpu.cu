#include <cuda_runtime.h>

// NOTE: if you include stdio.h, you can use printf inside your kernel

#include "common.h"
#include "matrix.h"
#include "mul_gpu.h"
#include <stdio.h>

// TODO (Task 4): Implement matrix multiplication CUDA kernel

__global__ void matrix_mul_kernel(float *m, float *n, float *p, int m_width, int n_width, int p_width, int p_height, size_t m_pitch, size_t n_pitch, size_t p_pitch)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tx >= p_width)
        return;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (ty >= p_height)
        return;
    float p_value = 0;

    for (int k = 0; k < m_width; k++)
    {
        // p_value += m[ty * m_width + k] * n[tx + k * n_width];
        float *m_row = (float*)((char*)m + ty * m_pitch);
        float m_element = m_row[k];

        float *n_row = (float*)((char*)n + k * n_pitch);
        float n_element = n_row[tx];

        p_value += m_element * n_element;
    }

    // p[ty * p_width + tx] = p_value;
    float *p_row = (float*)((char*)p + ty * p_pitch);
    p_row[tx] = p_value;
}

void matrix_mul_gpu(const GPUMatrix &m, const GPUMatrix &n, GPUMatrix &p)
{
	// TODO (Task 4): Determine execution configuration and call CUDA kernel
    int block_width = 32;
    int block_height = 32;
    dim3 dim_block(block_width, block_height);
	dim3 dim_grid(div_up(p.width, block_width), div_up(p.height, block_height));
	matrix_mul_kernel<<<dim_grid, dim_block>>>(m.elements, n.elements, p.elements, m.width, n.width, p.width, p.height, m.pitch, n.pitch, p.pitch);
}
