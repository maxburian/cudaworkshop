/*
 * Cooperative Groups example for JSC CUDA Course
 * 
 * -Andreas Herten, Apr 2018
 *
 */

#include <stdio.h>
#include <cuda.h>
#include <cooperative_groups.h>

#include <algorithm>

namespace cg = cooperative_groups;

// TODO: Make a CG thread group (cg::thread_group) the first argument of the function
__device__ int maxFunction(int * workspace, int value) {
	// TODO: Get the local rank (.thread_rank()) from the group, not from the global variable
	int lane = threadIdx.x;
	// TODO: Use the size of the group (.size()) and not the block dimension as the initial value of i
	for (int i = blockDim.x / 2; i > 0; i /= 2) {
		workspace[lane] = value;
		// TODO: Use the sync operation (.sync()) on the group rather than the intrinsic __syncthreads() function
		__syncthreads();

		if (lane < i)
			value = max(value, workspace[lane + i]);
		// TODO: Also here, use .sync() instead of __syncthreads()
		__syncthreads();
	}
	return value;
}

__global__ void maxKernel(int * array) {
	extern __shared__ int shmem_temp[];

	// TODO: Use this_thread_block() to create a CG thread block

	// TODO: Rather use the thread rank inside the group (.thread_rank()) to set threadIndex instead of the global variable threadIdx.x
	int threadIndex = threadIdx.x;
	int myValue = array[threadIndex];

	// TODO: The first argument of maxFunction should be this thread block from above
	int maxValue = maxFunction(shmem_temp, myValue);
	// TODO: Rather use the synchronization member function of the current thread block (.sync()) for synchronizing then the global intrinsic __syncthreads()
	__syncthreads();
	if (threadIndex == 0)
		array[0] = maxValue;
}

int main() {
	// This program finds the largest element in an array
	const int N = 128;
	int * array;
	cudaMallocManaged(&array, sizeof(int) * N);

	srand(42);

	for (int i = 0; i < N; i++)
		array[i] = rand() % 1024;

	int host_max = *(std::max_element(array, array+N));

	int blocks = 1;
	int threads = N;

	maxKernel<<<blocks, threads, threads * sizeof(int)>>>(array);
	cudaDeviceSynchronize();

	printf("GPU Max:  %d\n", array[0]);
	printf("Host Max: %d\n", host_max);
	if (host_max == array[0]) {
		printf("SUCCESS! both values match!\n");
	} else {
		printf("FAIL! Something went wrong. Values don't match.\n");
	}

	cudaFree(array);
}
