#include <stdio.h>

void checkError (const char* action);

int main (int argc, char** argv)
{
  int retVal = 0;    		/* return value */
  int numElements = 1<<22; 	/* 4194304 */
  float runtime = 0.0f;
  double oneSecondInMs = 1000.0;
  int oneGigaByte = 1<<30;

  cudaEvent_t startEvent;      /* to start timer */
  cudaEvent_t endEvent;	       /* to stop timer */

  float* inputBufferH;
  float* outputBufferH;
  float* inputBufferD;
  float* outputBufferD;

  cudaSetDevice(0);
  checkError("Initialize CUDA device");
  
  // create events used for timing
  cudaEventCreate( &startEvent );
  checkError("Create CUDA event startEvent");
  cudaEventCreate( &endEvent );
  checkError("Create CUDA event endEvent");

  // Allocate host memory
  cudaMallocHost( (void**) &inputBufferH, numElements*sizeof(float) );
  checkError("Allocate pinned host memory inputBufferH");
  cudaMallocHost( (void**) &outputBufferH, numElements*sizeof(float) );
  checkError("Allocate pinned host memory outputBufferH");

  // Allocate device memory
  cudaMalloc( (void**) &inputBufferD, numElements*sizeof(float) );
  checkError("Allocate device memory inputBufferD");
  cudaMalloc( (void**) &outputBufferD, numElements*sizeof(float) );
  checkError("Allocate device memory outputBufferD");

  // start timing
  cudaEventRecord ( startEvent );
  checkError("Record CUDA event startEvent");
	
  // copy input buffer to device and output buffer to host
  cudaMemcpy( inputBufferD, inputBufferH, numElements*sizeof(float), cudaMemcpyHostToDevice );
  checkError("Copy host to device");
  cudaMemcpy( outputBufferH, outputBufferD, numElements*sizeof(float), cudaMemcpyDeviceToHost );
  checkError("Copy device to host");

  // stop timer and get elapsed runtime for data transfers
  cudaEventRecord ( endEvent );
  checkError("Record CUDA event endEvent");
  cudaEventSynchronize ( endEvent );
  checkError("Synchronize CUDA event endEvent");
  cudaEventElapsedTime ( &runtime, startEvent, endEvent );
  checkError("Get Elapsed Time startEvent endEvent");
	
  printf("Bidirectional Bandwidth %f (GB/s)\n", numElements*sizeof(float)*oneSecondInMs / ( oneGigaByte * runtime ) );
	
  // Free host and device memory
  cudaFree( outputBufferD );
  cudaFree( inputBufferD );

  cudaFreeHost( outputBufferH );
  cudaFreeHost( inputBufferH );

  cudaEventDestroy( endEvent );
  cudaEventDestroy( startEvent );
  cudaDeviceSynchronize();
  cudaDeviceReset();

  return retVal;
}

/* Simple error checking function for CUDA actions */
void checkError (const char* action) {
  cudaError_t error;
  error = cudaGetLastError(); 

  if (error != cudaSuccess) {
    printf ("\nError while '%s': %s\nprogram terminated ...\n\n", action, cudaGetErrorString(error));
    exit (EXIT_SUCCESS);
  }
}
