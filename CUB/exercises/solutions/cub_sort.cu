#include <iostream>
#include <curand.h>
#include <cub/cub.cuh>

template <typename T>
__global__ void sort(const T* data_in, T* data_out){
    
    typedef cub::BlockRadixSort<T, 1024, 4> BlockRadixSortT;
    __shared__ typename BlockRadixSortT::TempStorage tmp_sort;
    
    
    double items[4];
    int i0 = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    for (int i = 0; i < 4; ++i){
        items[i] = data_in[i0 + i];
    }
    
    BlockRadixSortT(tmp_sort).Sort(items);
    
    for (int i = 0; i < 4; ++i){
        data_out[i0 + i] = items[i];
    }
}

int main(){
    double* d_gpu = NULL;
    double* result_gpu = NULL;
    double* data_sorted = new double[4096];
    // Allocate memory on the GPU
    cudaMalloc(&d_gpu, 4096 * sizeof(double));
    cudaMalloc(&result_gpu, 4096 * sizeof(double));
    
    curandGenerator_t gen;
    //     Create generator
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    //     Fill array with random numbers
    curandGenerateNormalDouble(gen, d_gpu, 4096, 0.0, 1.0);
    //     Destroy generator
    curandDestroyGenerator(gen);
    
    // Sort data    
    sort<<<1, 1024>>>(d_gpu, result_gpu);
    
    cudaMemcpy(data_sorted, result_gpu, 4096 * sizeof(double), cudaMemcpyDeviceToHost);
    // Write the sorted data to standard out
    
    for (int i = 0; i < 4096; ++i){
        std::cout << data_sorted[i] << ", ";
    }
    std::cout << std::endl;
}
