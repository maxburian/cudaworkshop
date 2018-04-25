/*
 * Cuda_MM_shared_memcpy.cu
 *
 * compile with: 
 *    -> nvcc -O3 -arch=sm_37 -o Cuda_MM_shared_memcpy Cuda_MM_shared_memcpy.cu
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
#ifdef _OPENMP
#include "omp.h"
#endif
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
    
  //       Allocate shared memory for the two blocks aSub and bSub.
  //       Use two-dimensional matrices of size BLOCK_SIZE * BLOCK_SIZE 
  __shared__ double aSub[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ double bSub[BLOCK_SIZE][BLOCK_SIZE];

  // Calculate global thread index 
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
      // Copy block into shared memory
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

	
    __syncthreads(); // Make sure that all threads had time to process input data.

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
  int size = n * n * sizeof(double);
  int xGrid, yGrid;
  double *d_a, *d_b, *d_c;
  float time;

  // Define events for timing
  cudaEvent_t start, stop;
  
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
    
  
  // Allocate memory for d_a, d_b and d_c on device
  cudaMalloc((void**) &d_a, size);
  checkError("cudaMalloc: d_a");
  cudaMalloc((void**) &d_b, size);
  checkError("cudaMalloc: d_b");  
  cudaMalloc((void**) &d_c, size);
  checkError("cudaMalloc: d_c");
    
    
  // First calculate grid size by dividing n by BLOCK_SIZE = 16
  xGrid = (n % BLOCK_SIZE == 0) ? (n / BLOCK_SIZE) : (n / BLOCK_SIZE + 1);
  yGrid = (n % BLOCK_SIZE == 0) ? (n / BLOCK_SIZE) : (n / BLOCK_SIZE + 1);
  dim3 gridDim(xGrid, yGrid);
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

  printf("\nGrid: %d, %d; block:%d, %d\n", xGrid , yGrid , BLOCK_SIZE, BLOCK_SIZE);
    
  // Invoke kernel and measure execution time 
  cudaEventRecord( start, 0 );
    
  // Copy data for a and b from host to device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice); 
  checkError("copying data of A from host to device");
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice); 
  checkError("copying data of B from host to device");
    
  
  // Call the kernel 
  dgemm_gpu_shared<<<gridDim, blockDim>>>(d_a, d_b, d_c, n);

  // Read restults from device memory to C 
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
  checkError("cudaMemcpyDeviceToHost");
    
  // Get elapsed time for kernel execution
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  checkError("executing Kernel");
  cudaEventElapsedTime( &time, start, stop );
  cudaEventDestroy( start );
  cudaEventDestroy( stop );

  // Free device memory 
  cudaFree(d_a);
  checkError("Freeing d_a");
  cudaFree(d_b);
  checkError("Freeing d_b");
  cudaFree(d_c);
  checkError("Freeing d_c");

  return time;

}

int main(int argc, char** argv)
{
  int n = 1024;
  double *a, *b, *c;
  int row, col;
  double absError, maxAbsError = 0.0, sumAbsError = 0.0;
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
  
    
  // Allocate memory for matrices on host
  assert ( a = (double*) malloc (n * n * sizeof(double)) );
  assert ( b = (double*) malloc (n * n * sizeof(double)) );
  assert ( c = (double*) malloc (n * n * sizeof(double)) );
    
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
  // Free memory on host
  free (a);
  free (b);
  free (c);
  
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