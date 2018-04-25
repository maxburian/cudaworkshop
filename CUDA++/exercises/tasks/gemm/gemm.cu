/*
 * gemm_gpu_shared.cu
 *
 * compile with: make 
 *
 * Matrices are stored as array in row-major order: 
 * A[row][col] = A[row * N + col]
 *
 * Use shared memory to speed up the matrix multiplication. We can reuse
 * the memory if we load a block of the matrix and have a thread block 
 * calculate a sub matrix.
 */


#include <stdio.h>
#include <assert.h>
#include <cuda.h>

#define CUDA_CALL( call )\
{\
    cudaError_t err = call;\
    if ( cudaSuccess != err)\
        fprintf(stderr, "CUDA error for %s in %d of %s : %s.\n", #call , __LINE__ , __FILE__ \
                ,cudaGetErrorString(err));\
}

// Thread block size: BLOCK_SIZE * BLOCK_SIZE
#define BLOCK_SIZE 16

// Declaration of helper functions (see bottom of file for details)
float getGflops (int, float);

/*
 *  Matrix multiplication kernel called by matrixMulOnDevice() 
 */
// TODO: Convert the kernel into a template function with typename T. Replace all occurences of 
// double with T.
__global__ void gemm_gpu_shared(double* a, double* b, double* c, int n){
    // Allocate shared memory for the two blocks aSub and bSub.
    // Use two-dimensional matrices of size BLOCK_SIZE * BLOCK_SIZE 
    __shared__ double aSub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double bSub[BLOCK_SIZE][BLOCK_SIZE];
    
    int idxX = blockDim.x * blockIdx.x + threadIdx.x;
    int idxY = blockDim.y * blockIdx.y + threadIdx.y;
    
    // For the matrix multiplication, we need to multiply all the elements of 
    // the idxYth row of a with all the elements of the idXth column of b and 
    // sum up the results.
    double sum = 0;

    // Calculate global offset of upper left corner of thread block.
    int blockaY = blockIdx.y * BLOCK_SIZE;
    int blockbX = blockIdx.x * BLOCK_SIZE;

    for (int block = 0; block < gridDim.x; ++block){
        // Get the two sub matrices
        int blockaX = block * (BLOCK_SIZE);
        int blockbY = block * (BLOCK_SIZE);
        if (((blockaY + threadIdx.y) < n) && (blockaX + threadIdx.x) < n) {
	  aSub[threadIdx.y][threadIdx.x] = a[(blockaY + threadIdx.y) * n + blockaX + threadIdx.x];
        } else {
            aSub[threadIdx.y][threadIdx.x] = 0;
        }

        if (((blockbY + threadIdx.y) < n) && (blockbX + threadIdx.x) < n) {
            bSub[threadIdx.y][threadIdx.x] = b[(blockbY + threadIdx.y) * n + blockbX + threadIdx.x];
        } else {
            bSub[threadIdx.y][threadIdx.x] = 0;
        }
	
	__syncthreads(); // Make sure that all threads had time to read the sub matrix.
    
        if ((idxX < n) && (idxY < n))
        {
            for (int i=0; i < blockDim.x; ++i){ //assumes that we use square blocks
                sum += aSub[threadIdx.y][i] * bSub[i][threadIdx.x];
            }
        }

	__syncthreads();
	
    }
    if ((idxX < n) && (idxY < n)){    
        c[idxY * n + idxX] = sum;
    }
}

/*
 *  Matrix multiplication host function called by main() 
 */

// TODO: Convert the kernel into a template function with typename T. Replace all occurences of 
// double with T.
void matrixMulOnDevice(double* a, double* b, double* c, int n)
{
    int xGrid, yGrid;
    float time;

    // Define events for timing
    cudaEvent_t start, stop;
  
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    
    // First calculate grid size by dividing n by BLOCK_SIZE = 16
    xGrid = (n % BLOCK_SIZE == 0) ? (n / BLOCK_SIZE) : (n / BLOCK_SIZE + 1);
    yGrid = (n % BLOCK_SIZE == 0) ? (n / BLOCK_SIZE) : (n / BLOCK_SIZE + 1);
    dim3 gridDim(xGrid, yGrid);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    printf("Grid: %d, %d; block:%d, %d\n", xGrid , yGrid , BLOCK_SIZE, BLOCK_SIZE);
    
    // Invoke kernel and measure execution time 
    CUDA_CALL(cudaEventRecord( start, 0 ));

    gemm_gpu_shared<<<gridDim, blockDim>>>(a, b, c, n);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize()); 

    CUDA_CALL(cudaEventRecord( stop, 0 ));
    CUDA_CALL(cudaEventSynchronize( stop ));
    
    // Get elapsed time for kernel execution
    CUDA_CALL(cudaEventElapsedTime( &time, start, stop ));
    CUDA_CALL(cudaEventDestroy( start ));
    CUDA_CALL(cudaEventDestroy( stop ));

    printf ("\nKernel Execution Time: %f ms (dim C: %d * %d)", time, n, n);
    printf ("\nThis corresponds to: %4.4f GFLOPS", getGflops(n, time));
  
}

int main(int argc, char** argv)
{
    int n = 1024;
    double *a, *b, *c;
    int row, col;
    double absError, maxAbsError = 0.0, sumAbsError = 0.0;
    size_t size; 

    if (argc > 1) {
       n = atoi(argv[1]);
    }

    // show banner
    printf ("\n\n     Matrix-Multiplication \n");
    printf (    "     ==========================================\n");
    printf (  "\n     Simple GEMM implemantation on GPU");  

    // echo device data
    int idevice = 0;
    CUDA_CALL(cudaSetDevice(idevice));
    cudaDeviceProp dprops;
    CUDA_CALL(cudaGetDeviceProperties( &dprops, idevice ));
    printf ("\n     Device name = %s, with compute capability %d.%d \n", 
	    dprops.name, dprops.major, dprops.minor);
    printf (  "\n     Matrix size %d x %d", n, n);
  
    
    // Allocate memory for matrices (that can be accessed from host and device) 
    size = n * n * sizeof(double);
    CUDA_CALL(cudaMallocManaged(&a, size));
    CUDA_CALL(cudaMallocManaged(&b, size));
    CUDA_CALL(cudaMallocManaged(&c, size));

    // Init matrices A and B: A = E so result will be B
    #pragma omp parallel for private(row, col)
    for (row = 0; row < n; ++row){
      for (col = 0; col < n; col++){
	a[row * n + col] = (row == col) ? 1.0 : 0.0;
	b[row * n + col] = row * n + col;
      }
    }

    // do matrix multiplication on device
    matrixMulOnDevice(a, b, c, n);
     
    // Compare results
    for ( row = 0; row < n; ++row){
      for ( col = 0; col < n; ++col) {
	
	absError = fabs ( c[row * n + col] - b[row * n + col]);
	sumAbsError += absError;
	
	if (absError > maxAbsError)
	  maxAbsError = absError;
      }
    }
    // Free memory 
    CUDA_CALL(cudaFree(a));
    CUDA_CALL(cudaFree(b));
    CUDA_CALL(cudaFree(c));

  
    printf ("\nmaxAbsError: %4.4f, sumAbsError: %4.4f", maxAbsError, sumAbsError);
    if (maxAbsError < 2.0e-5)
      printf ("\n\nProgram terminated SUCCESSFULLY.\n\n");

    return 0;
}

/*
 *  Some helper functions
 */

// get compute performance
float getGflops (int n, float time) {
    float gf = (2.0e-6 * n * n* n / time);
    return gf;
}
