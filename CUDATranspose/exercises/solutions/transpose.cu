#include <iostream>

const int BLOCK_SIZE = 16;
typedef long long Integer;

#define CUDA_CALL( call )                                                                                                           \
{                                                                                                                                   \
    cudaError_t result = call;                                                                                                      \
    if ( cudaSuccess != result )                                                                                                    \
        std::cerr<<"CUDA error "<< cudaGetErrorString( result )<<"("<<result<<") at line "<<__LINE__<<" of "<<__FILE__<<std::endl;  \
}

__global__ void transpose( Integer* const a_trans, const Integer* const a,  const Integer n )
{
    const Integer col_block = blockIdx.x;
    const Integer row_block = blockIdx.y;
    const Integer block_col = threadIdx.x;
    const Integer block_row = threadIdx.y;
    const Integer col = col_block*BLOCK_SIZE+block_col;
    const Integer row = row_block*BLOCK_SIZE+block_row;
    
    
    __shared__ Integer a_tile[BLOCK_SIZE][BLOCK_SIZE];
    
    if ( row < n && col < n )
    {
        a_tile[block_row][block_col] = a[row*n+col];
        __syncthreads();
    
        a_trans[(col_block*BLOCK_SIZE+block_row) * n + (row_block*BLOCK_SIZE+block_col)] = a_tile[block_col][block_row];
    
    }
}

int main()
{
    const Integer n = 8192;
    Integer* a = new Integer[n*n];
    
    for ( Integer row = 0; row < n; ++row )
    {
        for ( Integer col = 0; col < n; ++col )
        {
            a[row*n+col] = row*n+col;
        }
    }
    
    Integer* a_d;
    Integer* a_trans_d;
    
    CUDA_CALL( cudaMalloc( (void**) &a_d, n * n * sizeof( Integer ) ) );
    CUDA_CALL( cudaMalloc( (void**) &a_trans_d, n * n * sizeof( Integer ) ) );
    
    CUDA_CALL( cudaMemcpy( a_d, a, n * n * sizeof( Integer ), cudaMemcpyHostToDevice ) );
    
    cudaEvent_t start, stop;
    CUDA_CALL( cudaEventCreate( &start ) );
    CUDA_CALL( cudaEventCreate( &stop ) );
    
    const dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    const dim3 dimGrid((n%dimBlock.x) == 0 ? n/dimBlock.x : n/dimBlock.x+1, (n%dimBlock.y) == 0 ? n/dimBlock.y : n/dimBlock.y+1 );
    
    //warm up call
    transpose<<<dimGrid,dimBlock>>>( a_trans_d, a_d, n );
    
    CUDA_CALL( cudaDeviceSynchronize() );
    CUDA_CALL( cudaEventRecord(start) );
    transpose<<<dimGrid,dimBlock>>>( a_trans_d, a_d, n );
    CUDA_CALL( cudaGetLastError() );
    CUDA_CALL( cudaDeviceSynchronize() );
    CUDA_CALL( cudaEventRecord(stop) );
    
    CUDA_CALL( cudaMemcpy( a, a_trans_d, n * n * sizeof( Integer ), cudaMemcpyDeviceToHost ) );
    
    int errors = 0;
    for ( Integer row = 0; row < n; ++row )
    {
        for ( Integer col = 0; col < n; ++col )
        {
            if ( a[row*n+col] != col*n+row )
            {
                //Only print first error to avoid polluting the console
                if ( errors == 0 )
                {
                    std::cerr<<"First error at ("<<row<<","<<col<<") value is "<<a[row*n+col]<<" expected was "<<col*n+row<<std::endl;
                }
                ++errors;
            }
        }
    }
    
    if ( 0 == errors )
    {
        float runtime = 0.0f;
        cudaEventElapsedTime ( &runtime, start, stop );
        std::cout<<"Runtime: "<<runtime<<" ms, Mem BW "<<2*n*n*sizeof(Integer)/(1024*1024*1024*(runtime/1000.0f))<<" GB/s"<<std::endl;
    }
    
    cudaEventDestroy( stop );
    cudaEventDestroy( start );
    
    cudaFree( a_trans_d );
    cudaFree( a_d );
    delete[] a;
    return errors;
}
