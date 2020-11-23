#include <cmath>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <algorithm>

#include <cuda_runtime.h>

#include "common.h"
#include "matrix.h"


CPUMatrix matrix_alloc_cpu(int width, int height)
{
	CPUMatrix m;
	m.width = width;
	m.height = height;
	m.elements = new float[m.width * m.height];
	return m;
}
void matrix_free_cpu(CPUMatrix &m)
{
	delete[] m.elements;
}

GPUMatrix matrix_alloc_gpu(int width, int height)
{
	// TODO (Task 4): Allocate memory at the GPU
	GPUMatrix m;
	m.width = width;
	m.height = height;
	cudaMallocPitch((void**)&m.elements, &m.pitch, m.width * sizeof(float), m.height);
	return m;
}

void matrix_free_gpu(GPUMatrix &m)
{
	// TODO (Task 4): Free the memory
	cudaFree(m.elements);
}

void matrix_upload(const CPUMatrix &src, GPUMatrix &dst)
{
	// TODO (Task 4): Upload CPU matrix to the GPU
    cudaMemcpy2D(dst.elements, dst.pitch, src.elements, src.width * sizeof(float), src.width * sizeof(float), src.height, cudaMemcpyHostToDevice);
}

void matrix_download(const GPUMatrix &src, CPUMatrix &dst)
{
	// TODO (Task 4): Download matrix from the GPU
    cudaMemcpy2D(dst.elements, dst.width * sizeof(float), src.elements, src.pitch, src.width * sizeof(float), src.height, cudaMemcpyDeviceToHost);
}

void matrix_compare_cpu(const CPUMatrix &a, const CPUMatrix &b)
{
	// TODO (Task 4): compare both matrices a and b and print differences to the console
	float total_diff = 0.0f;
	int errors = 0;
	float diff;
	for (int i = 0; i < a.height * a.width; i++)
	{
		diff = std::fabs(a.elements[i] - b.elements[i]);
		total_diff += diff;
		if (diff < 10e-3)
			continue;
		errors++;
	}
	if (errors != 0)
		printf("\nResult matrices are different!");
	else
	{
		printf("\nResult matrices are identical with the total difference of %f.\n", total_diff);
	}
	
}
