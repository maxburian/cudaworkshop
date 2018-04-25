/*
 * CUDA Peer to Peer Example
 */
#include<cuda.h>
#include<stdio.h>
#include<sys/time.h>

#define SIZE 1048576
#define THREADS_PER_BLOCK 256
#define FLOAT(t) ((float)(t).tv_sec+((float)(t).tv_usec)/1000000)

#define CHECK_RUN( errorDescription ) { cudaError_t cerror; \
        if( (cerror = cudaGetLastError()) != cudaSuccess ){ \
            printf("execution aborted (function : %s, file : %s, line : %d) :\n    %s -> %s\n", \
                                       __func__, __FILE__, __LINE__, \
                                       errorDescription, \
                                       cudaGetErrorString(cerror) ); \
                                       return 0; } }

int run_copy_without_gpudirect(float *ha, float *tmp, float *hb, float *g0, float *g1, int size);                        
int run_copy_with_gpudirect(float *ha, float *tmp, float *hb, float *g0, float *g1, int size);
void check_results(float *a);

__global__ void kernel_gpu0 (float *g0);
__global__ void kernel_gpu1 (float *g1);

int threadsPerBlock, blocksPerGrid;

int main()
{

	struct timeval before, after, t1, t2;
	
	int size = SIZE*sizeof(float);
    	threadsPerBlock = THREADS_PER_BLOCK;
    	blocksPerGrid   = SIZE / threadsPerBlock;
	
	// Allocate memory on the host
	float *ha, *hb, *tmp;
	ha = (float*)malloc(size);
	hb = (float*)malloc(size);
	tmp = (float*)malloc(size);
	
	printf("Everything ok until here\n");	
	// Initialize the data
	int i;
	for(i=0; i<SIZE; i++)
	{
		ha[i] = 0;
		hb[i] = 0;
		tmp[i] = 0;
	}

	// Allocate memory on both the devices and enable peer access
	float *g0, *g1;
	cudaSetDevice(0);
	CHECK_RUN("Set Device");
	cudaSetDevice(0);
	CHECK_RUN("Set Device");
	cudaDeviceEnablePeerAccess(1, 0);		// PeerGPU, flags
	CHECK_RUN("Enable Peer Access");
	cudaMalloc(&g0, size);
	CHECK_RUN("Alloc g0");
	cudaSetDevice(1);
	CHECK_RUN("Set Device");
	cudaDeviceEnablePeerAccess(0, 0);		// PeerGPU, flags
	CHECK_RUN("Enable Peer Access");
	cudaMalloc(&g1, size);
	CHECK_RUN("Alloc g1");
			
	// Copy Data Without GPUdirect
	cudaSetDevice(0);
	CHECK_RUN("Set Device");
	gettimeofday(&before, NULL);
	run_copy_without_gpudirect(ha, tmp, hb, g0, g1, size);
	gettimeofday(&after, NULL);
	timersub(&after, &before, &t1);

	printf("Time without GPUdirect: %0.6f ms\n", FLOAT(t1)*1000);
	
	// Check results	
	check_results(hb);	

	// Copy Data With GPUdirect
	cudaSetDevice(0);
	CHECK_RUN("Set Device");
	gettimeofday(&before, NULL);
	run_copy_with_gpudirect(ha, tmp, hb, g0, g1, size);
	gettimeofday(&after, NULL);
	timersub(&after, &before, &t2);
	
	printf("Time with GPUdirect: %0.6f ms\n", FLOAT(t2)*1000);

	// Check results
	check_results(hb);

	// Free host memory
	free(ha);
	free(hb);
	free(tmp);

	// Free memory and disable peer access
	cudaSetDevice(0);
	CHECK_RUN("Set Device");
	cudaDeviceDisablePeerAccess(1);
	CHECK_RUN("Disable Peer Access");
	cudaFree(g0);
	CHECK_RUN("Free g0");
	cudaSetDevice(1);
	CHECK_RUN("Set Device");
	cudaDeviceDisablePeerAccess(0);
	CHECK_RUN("Disable Peer Access");
	cudaFree(g1);
	CHECK_RUN("Free g1");
	
	return 0;

}

int run_copy_without_gpudirect(float *ha, float *tmp, float *hb, float *g0, float *g1, int size)
{
	/* TODO: Do the following here 
	 * 1. Copy ha to g0 and run kernel_gpu0 with g0
	 * 2. Modified g0 must be the input to kernel_gpu1. What do you do here?
	 * 2. Copy final result to hb
	 * Tip: Don't forget to insert cudaSetDevice(0|1) at the right places
	 */
	
	// ha --> g0
	cudaMemcpy(g0, ha, size, cudaMemcpyDefault);
	CHECK_RUN("Memcpy ha --> g0");
 
	// Run gpu1 kernel
	kernel_gpu0<<<blocksPerGrid, threadsPerBlock>>>(g0);
	cudaThreadSynchronize();
	
	// g0 --> tmp --> g1
	cudaMemcpy(tmp, g0, size, cudaMemcpyDefault);
	CHECK_RUN("Memcpy g0 --> tmp");
	cudaSetDevice(1);
	CHECK_RUN("Set Device");
	cudaMemcpy(g1, tmp, size, cudaMemcpyDefault);
	CHECK_RUN("Memcpy tmp --> g1");
	
	kernel_gpu1<<<blocksPerGrid, threadsPerBlock>>>(g1);
	//cudaThreadSynchronize();

	// g1 --> hb
	cudaMemcpy(hb, g1, size, cudaMemcpyDefault);
	CHECK_RUN("Memcpy g1 --> hb");

	return 0;
}


int run_copy_with_gpudirect(float *ha, float *tmp, float *hb, float *g0, float *g1, int size)
{

	/* TODO: Do the following here 
	 * 1. Copy ha to g0 and run kernel_gpu0 with g0
	 * 2. Modified g0 must be the input to kernel_gpu1. What do you do here?
	 * 2. Copy final result to hb
	 * Tip: Don't forget to insert cudaSetDevice(0|1) at the right places
	 */
	 
	// ha --> g0
	cudaMemcpy(g0, ha, size, cudaMemcpyDefault);
	CHECK_RUN("Memcpy ha --> g0");
	 
	// Run gpu0 kernel
	kernel_gpu0<<< blocksPerGrid, threadsPerBlock >>> (g0);
	 
	// g0 --> g1
	cudaMemcpy(g1, g0, size, cudaMemcpyDefault);
	CHECK_RUN("Memcpy g0 -> g1");	 

	// Run gpu1 kernel
	cudaSetDevice(1);
	CHECK_RUN("Set Device");
	kernel_gpu1<<< blocksPerGrid, threadsPerBlock >>> (g1);
	 
	// g1 --> hb
	cudaMemcpy(hb, g1, size, cudaMemcpyDefault);
	CHECK_RUN("Memcpy g1 -> hb");

	return 0;
}

__global__
void kernel_gpu0 (float *g0)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if( i < SIZE)
		if(g0[i] == 0)
			g0[i] += 1;
	
	return;

}

__global__ 
void kernel_gpu1 (float *g1)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(i < SIZE)
		if(g1[i] == 1)
			g1[i] += 1;
			
	return;
}

void check_results(float *hb)
{
	int i;
	for(i=0; i<SIZE; i++)
	{
		if(hb[i] != 2)
		{
			printf("Test Result Failed\n");
			return;
		}
		hb[i] = 0;
	}

	printf("Test Result Successful\n");
}