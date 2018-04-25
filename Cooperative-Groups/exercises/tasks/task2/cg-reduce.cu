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

	// TODO: Partition thisThreadBlock into partitions of size 16 (type: `cg::thread_block_tile<>`) using `cg::tiled_partition<>()`

	// TODO: Give each partition its own sub-range in shared memory by calculating an offset from 0, unique to each tiled partition. We will use this offset with pointer arithmetic in the device function call below
	//		 * Hint: Our offset should be 0 for the first partition, 16 for the second partition, 32 for the third partition…
	//		 * Hint 2: There are different ways to achieve this here
	//       * Hint 3: See end of file for two possibilities
	int shmemOffset = 0;

	// NOTE: In comparison to the previous task, the second argument now adds an offset to the shared memory array
	// TODO: Make your tiled partition the first argument of `maxFunction` and not the global block
	int maxValue = maxFunction(thisThreadBlock, shmem_temp + shmemOffset, myValue);
	thisThreadBlock.sync();

	// NOTE: We now have local max values for each tile; we still need to find the max value amongst them
	// TODO: Open `if` no on the index `threadIndex` of the global group being 0 but on the index of the local partition being 0
	if (threadIndex == 0) {
		// TODO: Use atomicMax(*address, value) to compare if either `value` or the value at `address` is the maximum and write back the result to `address`. This is the thread-safe version of max(a, b);
		array[0] = maxValue;  // This includes race conditions when called from multiple threads
	}
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




// Offset Calculation: Solution
// 	* Possibility 1): int shmemOffset = thisThreadBlock.thread_rank() - tiledPartition.thread_rank(); 
//  * Possibility 2): int shmemOffset = 16 * int(thisThreadBlock.thread_rank() / 16); 
