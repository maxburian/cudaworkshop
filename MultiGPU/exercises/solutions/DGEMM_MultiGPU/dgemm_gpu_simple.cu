/*
 * dgemm_gpu_simple.cu
 *
 * compile with: nvcc -O3 -arch=sm_20 -o dgemm_gpu_simple dgemm_gpu_simple.cu -lcudart
 *
 * Matrices are stored as array in row-major order: 
 * A[row][col] = A[row * N + col]
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>

// Thread block size: BLOCK_SIZE * BLOCK_SIZE
#define BLOCK_SIZE 16 

#define TIMETOFLOAT(t) ((float)(t).tv_sec+((float)(t).tv_usec)/1000000)

// Declaration of helper functions (see below for details)
void checkError (const char* action);
float getGflops (int, float);


/*
 *  Matrix multiplication kernel called by matrixMulOnDevice()
 */
__global__ void dgemm_gpu_simple (double *a, double *b, double *c, int n) { 

  
  double Cvalue = 0.0; 

  // Get global threadId in x and y direction
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int col = blockIdx.x * blockDim.x + threadIdx.x; 


  // Each thread computes one element of C 
  // by accumulating results into Cvalue
  if ( row < n && col < n) {
    for (int e = 0; e < n; ++e) 
      Cvalue += a[row * n + e] * b[e * n + col]; 

    c[row * n + col] = Cvalue; 
  }
}


/*
 *  Matrix multiplication host function called by main() 
 */

void matrixMulOnDevice(const double *a, const double* b, double *c, int n) { 
  
  double *d_a;             // matirx A in device memory
  double *d_b;             // matirx B in device memory
  double *d_c;             // matirx C in device memory
  size_t size;
  struct timeval start, end, total;

  // Define grid and block layout for kernel execution
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE); 
  dim3 gridDim( (n + BLOCK_SIZE - 1) / blockDim.x, (n + BLOCK_SIZE - 1) / blockDim.y); 
  
  size = n * n * sizeof (double);

  // Allocate memory for d_a, d_b and d_c on device
  cudaMalloc((void**)&d_a, size);
  checkError("allocating device memory for A");
  cudaMalloc((void**)&d_b, size);
  checkError("allocating device memory for B");
  cudaMalloc((void**)&d_c, size);
  checkError("allocating device memory for C");

  // Copy data for a and b from host to device
  gettimeofday(&start, NULL);
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice); 
  checkError("copying data of A from host to device");
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice); 
  checkError("copying data of B from host to device");
  
  printf("Grid: %d, %d; block:%d, %d\n", gridDim.x, gridDim.y , BLOCK_SIZE, BLOCK_SIZE);
    
  dgemm_gpu_simple<<<gridDim, blockDim>>>(d_a, d_b, d_c, n); 

  // Read restults from device memory to C 
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost); 
  checkError("copying results from device to host");

  gettimeofday(&end, NULL);
  timersub(&end, &start, &total);
  printf ("\nExecution Time: %f ms (dim C: %d * %d)", TIMETOFLOAT(total), n, n);
  
  // Free device memory 
  cudaFree(d_a);
  checkError("Freeing d_a");
  cudaFree(d_b);
  checkError("Freeing d_b");
  cudaFree(d_c);
  checkError("Freeing d_c");

} 


/*
 *  Main program
 */
int main (int argc, char* argv[]) {

  int n = 1024; // dimension of square matrices
  double *a, *b, *c;
  int row, col;
  double absError, maxAbsError = 0.0, sumAbsError = 0.0;
  
  if (argc > 1) {
       n = atoi(argv[1]);
  }

  
  // Allocate memory for matrices on host
  assert ( a = (double*) malloc (n * n * sizeof(double)) );
  assert ( b = (double*) malloc (n * n * sizeof(double)) );
  assert ( c = (double*) malloc (n * n * sizeof(double)) );
  
  // Init matrices A and B
  #pragma omp parallel for
  for ( row = 0; row < n; row++ ) {
    for ( col = 0; col < n; col++ ) {
      a[row * n + col] = (row == col) ? 1.0 : 0.0;
      b[row * n + col] = row * n + col;
    }
  }

  // Execute matrix multiplication (on device and on host for reference
  matrixMulOnDevice (a, b, c, n);
  
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
  if (maxAbsError < 2.0e-5)
    printf ("\n\nProgram terminated SUCCESSFULLY.\n\n");

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
