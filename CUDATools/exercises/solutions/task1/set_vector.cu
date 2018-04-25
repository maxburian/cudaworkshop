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



__global__ void set(const int n, float* __restrict__ const a_d, const float value) {
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if ( i < n ) {
		a_d[ i ] = value;
	}
}


int main() {
	int n = 1024;

	CUDA_CHECK_RETURN(cudaSetDevice(0));

	float *a_d = 0;
	CUDA_CHECK_RETURN(cudaMalloc((void**) &a_d, n * sizeof(float)));

	float value = 3.14f;
	set<<<n/256,256>>>(n, a_d, value);
	CUDA_CHECK_RETURN(cudaGetLastError());

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	CUDA_CHECK_RETURN(cudaFree(a_d));
	CUDA_CHECK_RETURN(cudaDeviceReset());

	return 0;
}
