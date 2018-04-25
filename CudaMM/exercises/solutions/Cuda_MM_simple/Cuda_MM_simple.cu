/*
 * Cuda_MM_simple.cu
 *
 * compile with: make Cuda_MM_simple
 * run with:     make runGPU
 *
 * Matrices are stored as array in row-major order: 
 * A[row][col] = A[row * N + col]
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>

// Thread block size: BLOCK_SIZE * BLOCK_SIZE
#define BLOCK_SIZE 16 


// Declaration of helper functions (see below for details)
void checkError (const char* action);
float getGflops (int, float);


/*
 *  Matrix multiplication kernel called by matrixMulOnDevice()
 */
__global__ void dgemm_gpu_simple (const double *a, const double *b, double *c, int n) { 

  
  double Cvalue = 0.0; 

  // Get global threadId in x and y direction
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Each thread computes one element of C by accumulating results into Cvalue
  // Matrices are stored in row-major order: 
  // A[row][col] = A[row * N + col]
  // Remember to check bounds !
  if ( row < n && col < n) {
    for (int e = 0; e < n; ++e) 
      Cvalue += a[row * n + e] * b[e * n + col]; 

    c[row * n + col] = Cvalue; 
  }
}


/*
 *  Matrix multiplication host function called by main() 
 */

float matrixMulOnDevice(const double *a, const double* b, double *c, int n) { 
  
  float time;

  // Define grid and block layout for kernel execution
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE); 
  dim3 gridDim( (n + BLOCK_SIZE - 1) / blockDim.x, (n + BLOCK_SIZE - 1) / blockDim.y); 
  
  // Define events for timing
  cudaEvent_t start, stop; 
  
  cudaEventCreate(&start); 
  cudaEventCreate(&stop); 

  printf("Grid: %d, %d; block:%d, %d\n", gridDim.x, gridDim.y , BLOCK_SIZE, BLOCK_SIZE);
    
  // Invoke kernel and measure execution time 
  cudaEventRecord(start, 0);

  dgemm_gpu_simple<<<gridDim, blockDim>>>(a, b, c, n);  
  cudaDeviceSynchronize();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  checkError("kernel execution");

  // Get elapsed time for kernel execution
  cudaEventElapsedTime(&time, start, stop); 
  cudaEventDestroy(start); 
  cudaEventDestroy(stop);

  return time;
} 


/*
 *  Main program
 */
int main (int argc, char* argv[]) {

  int n = 1024; // dimension of square matrices
  double *a, *b, *c;
  int row, col;
  double absError, maxAbsError = 0.0, sumAbsError = 0.0;
  size_t size ;
  float time;

  if (argc > 1) {
       n = atoi(argv[1]);
  }

  
  // show banner
  printf ("\n\nMatrix-Multiplication \n");
  printf (    "==========================================\n");
  printf (  "\nSimple MM implemantation on GPU");  

  // echo device data
  int idevice = 0;
  cudaSetDevice(idevice);
  cudaDeviceProp dprops;
  cudaGetDeviceProperties( &dprops, idevice );
  printf ("\nDevice name = %s, with compute capability %d.%d \n", 
	  dprops.name, dprops.major, dprops.minor);
  printf (  "\nMatrix size %d x %d\n", n, n);
  
  // Allocate memory for matrices using UVM (can be accessed by host and device)
  size = n * n * sizeof(double);
  cudaMallocManaged((void**)&a, size);
  checkError("allocating UVM memory for A");
  cudaMallocManaged((void**)&b, size);
  checkError("allocating UVM memory for B");
  cudaMallocManaged((void**)&c, size);
  checkError("allocating UVM memory for C");
  // Init matrices A and B
  #pragma omp parallel for
  for ( row = 0; row < n; row++ ) {
    for ( col = 0; col < n; col++ ) {
      a[row * n + col] = (row == col) ? 1.0 : 0.0;
      b[row * n + col] = row * n + col;
    }
  }

  // Execute matrix multiplication (on device and on host for reference
  time = matrixMulOnDevice (a, b, c, n);
  
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
  cudaFree (a);
  cudaFree (b);
  cudaFree (c);
  
  printf ("\nmaxAbsError: %4.4f, sumAbsError: %4.4f", maxAbsError, sumAbsError);
  if (maxAbsError < 2.0e-5) {
    printf ("\nProgram terminated SUCCESSFULLY.\n");
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

// Simple error checking function for CUDA actions
void checkError (const char* action) {
  
  cudaError_t error;
  error = cudaGetLastError(); 

  if (error != cudaSuccess) {
    printf ("\nError while '%s': %s\nprogram terminated ...\n\n", action, cudaGetErrorString(error));
    exit (EXIT_SUCCESS);
  }
}

// Compute reference results on host
void dgemm_cpu_simple (const double* a, const double* b, double* c, int n) {
  
  int row, col, k;    // loop variables
  double val;         // help variable for results
  
  /*
    PERFORM MULTIPLICATION
  */
  // loop over output rows
#pragma omp parallel for
  for ( row=0; row<n; row++ ) {
    
    // loop over output columns
    for ( col=0; col<n; col++ ) {
      
      // initialize output result to zero
      val = 0;
      
      // loop over inner dimension
      for ( k=0; k<n; k++ ) {
        // sum
        val += a[row*n+k] * b[k*n+col];
      }
      c[row*n+col] = val;
    }
  }
}

// Print the values of a matrix on the screen
// could be useful for debugging
void printMatrix (const double* m, int n) {
  
  int i, j;
  
  for (i = 0; i < n; ++i) {
    printf("\n");
    for ( j = 0; j < n; ++j) {
      printf("%6.3f", m[i * n + j]);
    }
  }
}


// get compute performance
float getGflops (int width, float time) {

	float gf = (2.0e-6 * width * width* width / time);

	return gf;
}
