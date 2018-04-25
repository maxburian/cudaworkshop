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

// NOTE: The group used is a `cg::thread_block_tile<>` rather than `cg::thread_group`
// NOTE: We use templates now to specify the size of the group at compile time
// TODO: Remove shared memory `int * workspace` from function definition
template <int tile_size>
__device__ int maxFunction(cg::thread_block_tile<tile_size> g, int * workspace, int value) {
	// TODO: Remove lane; we don't need it any more
	int lane = g.thread_rank(); 
	for (int i = g.size() / 2; i > 0; i /= 2) {
		// TODO: Remove everything but the `value = max()` line of this loop. Instead of the shared memory `workspace[lane + i]` we use directly the value of the lane of the warp via the `.shfl_down()` function of the group.
		// HINT: We want the `value` of a lane which is `i` lanes away *further up* in the warp
		workspace[lane] = value;
		g.sync();

		if (lane < i)
			value = max(value, workspace[lane + i]);
		g.sync();
	}
	return value;
}

// NOTE: The kernel is equal to the kernel of task2 except for two changes:
//        1) The size of the static partition is a template parameter (`tile_size`)
//        2) Shared memory is not needed any more
template <int tile_size>
__global__ void maxKernel(int * array) {
	// TODO: Remove shared memory
	extern __shared__ int shmem_temp[];

	cg::thread_block thisThreadBlock = cg::this_thread_block();

	int threadIndex = thisThreadBlock.thread_rank();
	int myValue = array[threadIndex];

	cg::thread_block_tile<tile_size> tiledPartition = cg::tiled_partition<tile_size>(thisThreadBlock);

	// TODO: We don't need shared memory any more, remove this line
	int shmemOffset = thisThreadBlock.thread_rank() - tiledPartition.thread_rank();

	// TODO: Remove shared memory to adapt to new `maxFunction` call
	int maxValue = maxFunction<tile_size>(tiledPartition, shmem_temp + shmemOffset, myValue);
	thisThreadBlock.sync();

	if (tiledPartition.thread_rank() == 0) {
		atomicMax(&array[0], maxValue);
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

	maxKernel<16><<<blocks, threads, threads * sizeof(int)>>>(array);
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
