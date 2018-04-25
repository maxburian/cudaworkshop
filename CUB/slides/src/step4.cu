#include <cub/cub.cuh>

__global__ void ExampleKernel(const double* in, double* out){

    // Specialize the template for double precision and 128 threads w/ 4 items per thread   
    typedef cub::BlockLoad<double*, 128, 4> BlockLoadT;
    // Specialize the template for double precision and 128 threads    
    typedef cub::BlockReduce<double, 128> BlockReduceT;
    // Declare shared storage
    
    __shared__ union {
        typename BlockLoadT::TempStorage load;
        typename BlockReduceT::TempStorage reduce;
    } temp_storage;
    
    double items[4];
    
    BlockLoadT(temp_storage.load).Load(in, items);
    __syncthreads();
    
    // Instantiate an instance of BlockReduceT
    double result = BlockReduceT(temp_storage.reduce).Sum(items);
    
    if (threadIdx.x == 0){
        *out = result;
    }
}
    
        