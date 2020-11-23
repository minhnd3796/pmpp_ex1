#include <iostream>

#include <cuda_runtime.h>

#include "matmul.h"
#include "test.h"
#include "common.h"
#include "mul_cpu.h"
#include "mul_gpu.h"
#include "timer.h"

void print_cuda_devices()
{
	// TODO: Task 2
    printf("\nTask 2\n");
    cudaDeviceProp deviceProp;
	int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    for (int dev = 0; dev < deviceCount; dev++)
	{
		cudaSetDevice(dev);
		cudaGetDeviceProperties(&deviceProp, dev);
		printf("Device %d: \"%s\"\n", dev, deviceProp.name);
		printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
		printf("Multiprocessor count: %d\n", deviceProp.multiProcessorCount);
		printf("GPU clock rate in GHz: %0.2f GHz\n", deviceProp.clockRate * 1e-6f);
		printf("Total global memory in MiB: %.0f\n", deviceProp.totalGlobalMem / 1048576.0f);
		printf("L2 cache size in KiB: %.0f\n\n", deviceProp.l2CacheSize / 1024.0f);
	}
}

void matmul()
{
	// === Task 3 ===
	// TODO: Allocate CPU matrices (see matrix.cc)
	//       Matrix sizes:
	//       Input matrices:
	//       Matrix M: pmpp::M_WIDTH, pmpp::M_HEIGHT
	//       Matrix N: pmpp::N_WIDTH, pmpp::N_HEIGHT
	//       Output matrices:
	//       Matrix P: pmpp::P_WIDTH, pmpp::P_HEIGHT
	// CPUMatrix h_M = matrix_alloc_cpu(pmpp::M_WIDTH, pmpp::M_HEIGHT);	
	CPUMatrix h_M = matrix_alloc_cpu(pmpp::M_WIDTH, pmpp::M_HEIGHT);
	CPUMatrix h_N = matrix_alloc_cpu(pmpp::N_WIDTH, pmpp::N_HEIGHT);
	CPUMatrix h_P = matrix_alloc_cpu(pmpp::P_WIDTH, pmpp::P_HEIGHT);
	
	// TODO: Fill the CPU input matrices with the provided test values (pmpp::fill(CPUMatrix &m, CPUMatrix &n))
	pmpp::fill(h_M, h_N);

	// TODO (Task 5): Start CPU timing here!
	timer_tp start_cpu = timer_now();

	// TODO: Run your implementation on the CPU (see mul_cpu.cc)
	printf("Task 3\n");
	matrix_mul_cpu(h_M, h_N, h_P);

	// TODO (Task 5): Stop CPU timing here!
	timer_tp end_cpu = timer_now();
	float elapsed_time_ms_cpu = timer_elapsed(start_cpu, end_cpu);
	printf("CPU processing took: %f ms\n", elapsed_time_ms_cpu);

	// TODO: Check your matrix for correctness (pmpp::test_cpu(const CPUMatrix &p))
	pmpp::test_cpu(h_P);


	// === Task 4 ===
	// TODO: Set CUDA device
	int dev = 0;
	cudaSetDevice(dev);

	// TODO: Allocate GPU matrices (see matrix.cc)
	GPUMatrix d_M = matrix_alloc_gpu(pmpp::M_WIDTH, pmpp::M_HEIGHT);
	GPUMatrix d_N = matrix_alloc_gpu(pmpp::N_WIDTH, pmpp::N_HEIGHT);
	GPUMatrix d_P = matrix_alloc_gpu(pmpp::P_WIDTH, pmpp::P_HEIGHT);
	
	// TODO: Upload the CPU input matrices to the GPU (see matrix.cc)
	matrix_upload(h_M, d_M);
	matrix_upload(h_N, d_N);


	// TODO (Task 5): Start GPU timing here!
	cudaEvent_t evStart, evStop;
	cudaEventCreate(&evStart);
	cudaEventCreate(&evStop);
	cudaEventRecord(evStart, 0);

	// TODO: Run your implementation on the GPU (see mul_gpu.cu)
	printf("\nTask 4\n");
	matrix_mul_gpu(d_M, d_N, d_P);

	// TODO (Task 5): Stop GPU timing here!
	cudaEventRecord(evStop, 0);
	cudaEventSynchronize(evStop);
	float elapsed_time_ms_gpu;
	cudaEventElapsedTime(&elapsed_time_ms_gpu, evStart, evStop);
	printf("CUDA processing took: %f ms\n", elapsed_time_ms_gpu);

	// TODO: Download the GPU output matrix to the CPU (see matrix.cc)
	CPUMatrix h_P2 = matrix_alloc_cpu(pmpp::P_WIDTH, pmpp::P_HEIGHT);
	matrix_download(d_P, h_P2);


	// TODO: Check your downloaded matrix for correctness (pmpp::test_gpu(const CPUMatrix &p))
	pmpp::test_gpu(h_P2);

	// TODO: Compare CPU result with GPU result (see matrix.cc)
	matrix_compare_cpu(h_P, h_P2);

	// TODO (Task3/4/5): Cleanup ALL matrices and and events
	matrix_free_cpu(h_M);
	matrix_free_cpu(h_N);
	matrix_free_cpu(h_P);
	matrix_free_cpu(h_P2);

	matrix_free_gpu(d_M);
	matrix_free_gpu(d_N);
	matrix_free_gpu(d_P);

	cudaEventDestroy(evStart);
	cudaEventDestroy(evStop);
}


/************************************************************
 * 
 * TODO: Write your text answers here!
 * 
 * (Task 4) 6. Where do the differences come from?
 * 
 * Answer: The GPU has fused multiply-add while the CPU does not. Parallelizing
 * algorithms may rearrange operations, yielding different numeric results. The
 * CPU may be computing results in a precision higher than expected. Finally,
 * many common mathematical functions are not required by the IEEE 754 standard
 * to be correctly rounded so should not be expected to yield identical results
 * between implementations.
 * 
 * Source: Floating Point and IEEE 754 Compliance for NVIDIA GPUs - CUDA
 * Toolkit Documentation
 * https://docs.nvidia.com/cuda/floating-point/index.html#verifying-gpu-results
 * 
 * 
 ************************************************************/
