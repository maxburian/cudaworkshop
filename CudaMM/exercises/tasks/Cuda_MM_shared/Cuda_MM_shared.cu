/*
 * Cuda_MM_shared.cu
 *
 * compile with: make Cuda_MM_shared_uvm
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

// Thread block size: BLOCK_SIZE * BLOCK_SIZE
#define BLOCK_SIZE 16

// Declaration of helper functions (see bottom of file for details)
void checkError (const char* action);
float getGflops (int, float);

/*
 *  Matrix multiplication kernel called by matrixMulOnDevice() 
 */
__global__ void dgemm_gpu_shared(double* a, double* b, double* c, int n){
    
    // Allocate shared memory for the two blocks aSub and bSub.
    // Use two-dimensional matrices of size BLOC_SIZE*BLOCK_SIZE 
    // TODO
    __shared__ double aSub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double bSub[BLOCK_SIZE][BLOCK_SIZE];

      
    // Calculate global thread index 
    // TODO
    int idxX = blockIdx.x * blockDim.x+threadIdx.x;
    int idxY = blockIdx.y * blockDim.y+threadIdx.y;

    // For the matrix multiplication, we need to multiply all the elements of 
    // the idxYth row of a with all the elements of the idXth column of b and 
    // sum up the results.
    double sum = 0;

    // Calculate global offset of upper left corner of thread block.
    // TODO
    int blockaY = blockIdx.y * blockDim.y;
    int blockbX = blockIdx.x * blockDim.x;

    for (int block = 0; block < gridDim.x; ++block){
        // Get the two sub matrices
        int blockaX = block * (BLOCK_SIZE);
        int blockbY = block * (BLOCK_SIZE);
        if (((blockaY + threadIdx.y) < n) && (blockaX + threadIdx.x) < n) {
          // Copy block into shared memory
	  // TODO
	  asub[threadIdx.y][threadIdx.x] = a[(blockaY+threadIdx.y)*n+blockbX+threadIdx.x]; 
        } else {
            aSub[threadIdx.y][threadIdx.x] = 0;
        }

        if (((blockbY + threadIdx.y) < n) && (blockbX + threadIdx.x) < n) {
            bSub[threadIdx.y][threadIdx.x] = b[(blockbY + threadIdx.y) * n + blockbX + threadIdx.x];
        } else {
            bSub[threadIdx.y][threadIdx.x] = 0;
        }
	
	// TODO: Synchronize threads to make sure all threads are done copying
	__syncthreads();

        if ((idxX < n) && (idxY < n))
        {
            for (int i=0; i < blockDim.x; ++i){ //assumes that we use square blocks
                sum += aSub[threadIdx.y][i] * bSub[i][threadIdx.x];
            }
        }
	// TODO: Synchronize threads to make sure all threads are done with the data
	__syncthreads();	

    }
    if ((idxX < n) && (idxY < n)){    
        c[idxY * n + idxX] = sum;
    }
}



/*
 *  Matrix multiplication host function called by main() 
 */

float matrixMulOnDevice(double* a, double* b, double* c, int n)
{
    int xGrid, yGrid;
    float time;

    // Define events for timing
    cudaEvent_t start, stop;
  
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // First calculate grid size by dividing n by BLOCK_SIZE = 16
    xGrid = (n % BLOCK_SIZE == 0) ? (n / BLOCK_SIZE) : (n / BLOCK_SIZE + 1);
    yGrid = (n % BLOCK_SIZE == 0) ? (n / BLOCK_SIZE) : (n / BLOCK_SIZE + 1);
    dim3 gridDim(xGrid, yGrid);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    printf("\nGrid: %d, %d; block:%d, %d\n", xGrid , yGrid , BLOCK_SIZE, BLOCK_SIZE);
    
    // Invoke kernel and measure execution time 
    cudaEventRecord( start, 0 );
    

    // Call the kernel 
    // TODO
    dgemm_gpu_shared<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize(); 

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    checkError("executing Kernel");
    
    // Get elapsed time for kernel execution
    cudaEventElapsedTime( &time, start, stop );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    return time;

}

int main(int argc, char** argv)
{
    int n = 1024;
    double *a, *b, *c;
    int row, col;
    double absError, maxAbsError = 0.0, sumAbsError = 0.0;
    size_t size; 
    float time;

    if (argc > 1) {
       n = atoi(argv[1]);
    }

    // show banner
    printf ("\n\nMatrix-Multiplication \n");
    printf (    "==========================================\n");
    printf (  "\nGPU implemantation using shared memory");  

    // echo device data
    int idevice = 0;
    cudaSetDevice(idevice);
    cudaDeviceProp dprops;
    cudaGetDeviceProperties( &dprops, idevice );
    printf ("\nDevice name = %s, with compute capability %d.%d \n", 
	    dprops.name, dprops.major, dprops.minor);
    printf (  "\nMatrix size %d x %d", n, n);
  
    
    // Allocate memory for matrices (that can be accessed from host and device) 
    size = n * n * sizeof(double);
    ...
    checkError("cudaMallocManaged: a");
    ...
    checkError("cudaMallocManaged: b");  
    ...
    checkError("cudaMallocManaged: c");

    // Init matrices A and B: A = E so result will be B
    #pragma omp parallel for private(row, col)
    for (row = 0; row < n; ++row){
      for (col = 0; col < n; col++){
	a[row * n + col] = (row == col) ? 1.0 : 0.0;
	b[row * n + col] = row * n + col;
      }
    }

    // do matrix multiplication on device
    time = matrixMulOnDevice(a, b, c, n);
     
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
    cudaFree(a);
    checkError("Freeing a");
    cudaFree(b);
    checkError("Freeing b");
    cudaFree(c);
    checkError("Freeing c");

  
    printf ("\nmaxAbsError: %4.4f, sumAbsError: %4.4f", maxAbsError, sumAbsError);
    if (maxAbsError < 2.0e-5) {
      printf ("\n\nProgram terminated SUCCESSFULLY.\n\n");
      printf ("\nKernel Execution Time: %f ms (dim C: %d * %d)", time, n, n);
      printf ("\nThis corresponds to: %4.4f GFLOPS\n\n", getGflops(n, time));
    } else {
      printf ("\n--> Result not correct:  check your code\n\n");
    }
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

// Simple error checking function for CUDA actions

void checkError (const char* action) {
  
  cudaError_t error;
  error = cudaGetLastError(); 

  if (error != cudaSuccess) {
    printf ("\nError while '%s': %s\nprogram terminated ...\n\n", action, cudaGetErrorString(error));
    exit (EXIT_FAILURE);
  }
}
