#include <cub/cub.cuh>

__global__ void ExampleKernel(...){

    // Specialize the template for double precision and 128 threads    
    typedef cub::BlockReduce<double, 128> BlockReduceT;
    
    // Declare shared storage
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    
    double items[4];
    
    ...
}