#include <iostream>
#include <curand.h>
#include <cub/cub.cuh>

template <int BLOCK_WIDTH, int ITEMS_PER_THREAD, 
          cub::BlockLoadAlgorithm BLOCK_LOAD_ALGO,
          cub::BlockStoreAlgorithm BLOCK_STORE_ALGO,
          typename T>
__global__ void sort(const T* data_in, T* data_out){
    
    typedef cub::BlockLoad<T, BLOCK_WIDTH, ITEMS_PER_THREAD, BLOCK_LOAD_ALGO> BlockLoadT;
    typedef cub::BlockRadixSort<T, BLOCK_WIDTH, ITEMS_PER_THREAD> BlockRadixSortT;
    typedef cub::BlockStore<T, BLOCK_WIDTH, ITEMS_PER_THREAD, BLOCK_STORE_ALGO> BlockStoreT;
    
    __shared__ union {
        typename BlockLoadT::TempStorage load;
        typename BlockRadixSortT::TempStorage sort;
        typename BlockStoreT::TempStorage store;
    } tmp_storage;
    
    
    T items[ITEMS_PER_THREAD];
    BlockLoadT(tmp_storage.load).Load(data_in + blockIdx.x * BLOCK_WIDTH * ITEMS_PER_THREAD, items);
    __syncthreads();

    BlockRadixSortT(tmp_storage.sort).Sort(items);
    __syncthreads();
    
    BlockStoreT(tmp_storage.store).Store(data_out + blockIdx.x * BLOCK_WIDTH * ITEMS_PER_THREAD, 
                items);
    
}

int main(){
    double* d_gpu = NULL;
    double* result_gpu = NULL;
    double* data_sorted = new double[1000*4096];
    // Allocate memory on the GPU
    cudaMalloc(&d_gpu, 1000*4096 * sizeof(double));
    cudaMalloc(&result_gpu, 1000*4096 * sizeof(double));
    
    curandGenerator_t gen;
    //     Create generator
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    //     Fill array with random numbers
    curandGenerateNormalDouble(gen, d_gpu, 1000*4096, 0.0, 1.0);
    //     Destroy generator
    curandDestroyGenerator(gen);
    
    // Sort data    
    sort<512, 8, cub::BLOCK_LOAD_TRANSPOSE, cub::BLOCK_STORE_TRANSPOSE><<<1000, 512>>>(d_gpu, 
                                                                                       result_gpu);
    
    sort<256, 16, cub::BLOCK_LOAD_DIRECT, cub::BLOCK_STORE_DIRECT><<<1000, 256>>>(d_gpu, 
                                                                                  result_gpu);
    cudaMemcpy(data_sorted, result_gpu, 1000*4096*sizeof(double), cudaMemcpyDeviceToHost);

    // Write the sorted data to standard out
    for (int i = 0; i < 4095; ++i){
        std::cout << data_sorted[i] << ", ";
    }
    
    std::cout << data_sorted[4095] << std::endl;
}
