#include <cstdio>

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }



__global__ void print_test() {
	int i = 0;
	printf("blockIdx.x = %d, threadIdx.x = %d, i = %d\n", blockIdx.x, threadIdx.x, i);
}


int main() {
	CUDA_CHECK_RETURN(cudaSetDevice(0));

	print_test<<<2,32>>>();
	CUDA_CHECK_RETURN(cudaGetLastError());

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	CUDA_CHECK_RETURN(cudaDeviceReset());

	return 0;
}
