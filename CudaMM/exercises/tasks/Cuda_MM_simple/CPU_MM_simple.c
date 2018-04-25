/******************************************************************************
* DGEMM simple CPU implementation 
* 
******************************************************************************/


// include headers
#include <stdlib.h>
#include <stdio.h>
#ifdef _OPENMP
#include "omp.h"
#endif

// Declaration of helper functions (see below for details)
float getGflops (int, float);

void dgemm_cpu_simple (double* a, double* b, double* c, int n) {
 
  int row, col, k;    // loop variables
  double val;         // help variable for results
  
   /*
     PERFORM MULTIPLICATION
  */
  // loop over output rows
#pragma omp parallel for private(row, col, val, k)
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

int main (int argc, char* argv[]) {

  int n = 1024;       // matrix dimension
  
  double *a ;         // matrices stored as row-major arrays  
  double *b ;         // matrices stored as row-major arrays  
  double *c ;         // matrices stored as row-major arrays  

  int row, col;    // loop variables
  
  if (argc > 1) {
       n = atoi(argv[1]);
  }

  /* 
     TODO: uncomment to activate Cuda timing
  */
  float time, start, end;
  
  // show banner
  printf ("\n\nMatrix-Multiplication \n");
  printf (    "==========================================\n");
  printf (  "\nSimple DGEMM implemantation on HOST"); 
  printf (  "\nMatrix size %d x %d", n, n);
  
  #pragma omp parallel
#ifdef _OPENMP
  if (omp_get_thread_num() == 0 )
    printf ( "\n     Using %d OMP threads\n", omp_get_num_threads());
#endif

  
  
  // allocate arrays and initialize data
  a = (double *) malloc ( n*n*sizeof(double) );
  b = (double *) malloc ( n*n*sizeof(double) );
  c = (double *) malloc ( n*n*sizeof(double) );

  #pragma omp parallel for private(row, col)
  for ( row = 0; row<n; row++ ) {
    for ( col = 0; col<n; col++ ) {
      // data is in row-major format
      a[row*n+col] = (row == col) ? 1.0 : 0.0;
      b[row*n+col] = row * n + col;
    }
  }

  /* 
     TODO: uncomment to activate Cuda timing
  */
  
  start = omp_get_wtime();
  dgemm_cpu_simple (a, b, c, n);
  end = omp_get_wtime();
 
  
  // Get elapsed time for kernel execution
  time = end - start; 
  printf ("\nGFLOPS: %4.4f\n", getGflops(n, time));
  
  

  // write reference element for debugging
  //printf("\n     reference element c(768,768) = %4.4f <->  %4.4f\n", c[768*n+768], b[768*n+768] ); 
  
  // free memory
  free(a);
  free(b);
  free(c);

  return 0;

}


// get compute performance
float getGflops (int n, float time) {

  float gf = (2.0e-9 * n * n * n / time);
  return gf;

}
