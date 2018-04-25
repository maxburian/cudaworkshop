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

__device__ int maxFunction(cg::thread_group g, int * workspace, int value) {
	int lane = g.thread_rank(); 
	for (int i = g.size() / 2; i > 0; i /= 2) {
		workspace[lane] = value;
		g.sync();

		if (lane < i)
			value = max(value, workspace[lane + i]);
		g.sync();
	}
	return value;
}

__global__ void maxKernel(int * array) {
	extern __shared__ int shmem_temp[];

	cg::thread_block thisThreadBlock = cg::this_thread_block();

	int threadIndex = thisThreadBlock.thread_rank();
	int myValue = array[threadIndex];

	int maxValue = maxFunction(thisThreadBlock, shmem_temp, myValue);
	thisThreadBlock.sync();
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
