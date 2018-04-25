#include <stdio.h>
#include <stdlib.h>
#include "cublas_v2.h"

void checkError (const char* action);

int main (int argc, char** argv)
{
  int retVal = 0;    			/* return value */
  int numElements = 1<<22; 	/* 4194304 */
  float alpha = 5.0f;
  float runtime = 0.0f;
  double oneSecondInMs = 1000.0;
  int oneGigaByte = 1<<30;
  int i;
  float absError, maxAbsError = 0.0, sumAbsError = 0.0;

  cudaEvent_t startEvent;
  cudaEvent_t endEvent;

  cudaStream_t stream;
  cublasStatus_t cuBLASstat;
  cublasHandle_t cuBLAShandle;

  float* xH;
  float* yH;
  float* yDeviceResultH; // needed, since host will store its result to yH
  float* xD;
  float* yD;
  
  cudaSetDevice(0);
  checkError("Initialize CUDA device");

  // Set up events and stream to be used
  cudaEventCreate( &startEvent );
  checkError("Create CUDA event startEvent");
  cudaEventCreate( &endEvent );
  checkError("Create CUDA event endEvent");
  cudaStreamCreate( &stream );
  checkError("Create CUDA stream");
	
  // set up cuBLAS handle
  cuBLASstat = cublasCreate(&cuBLAShandle);
  if (cuBLASstat != CUBLAS_STATUS_SUCCESS) {
    printf ("CUBLAS initialization failed\n");
    return EXIT_FAILURE;
  }
	
  // allocate (pinned) host and device memory 
  cudaMallocHost( (void**) &xH, numElements*sizeof(float) );
  checkError("Allocate pinned host memory xH");
  cudaMallocHost( (void**) &yH, numElements*sizeof(float) );
  checkError("Allocate pinned host memory yH");
  cudaMallocHost( (void**) &yDeviceResultH, numElements*sizeof(float) );
  checkError("Allocate pinned host memory yDeviceResultH");
  cudaMalloc( (void**) &xD, numElements*sizeof(float) );
  checkError("Allocate device memory xD");
  cudaMalloc( (void**) &yD, numElements*sizeof(float) );
  checkError("Allocate device memory yD");
	
  //Initilaize host data
  srand(124);
#pragma omp parallel for
  for (i=0; i<numElements; ++i)
    {
      xH[i] = rand() / (float)RAND_MAX;
      yH[i] = rand() / (float)RAND_MAX;
    }

  //copy input to device
  cudaMemcpyAsync( xD, xH, numElements*sizeof(float), cudaMemcpyHostToDevice, stream );
  checkError("Start copy host to device for x");
  cudaMemcpyAsync( yD, yH, numElements*sizeof(float), cudaMemcpyHostToDevice, stream );
  checkError("Start copy host to device for y");
	
  //TODO: Set cuBLAS execution stream
  //cudaStream_t stream;
  cudaStreamCreate( &stream);
  cudaEventCreate(&startEvent);
  
  cudaEventRecord ( startEvent, stream );
  checkError("Record CUDA event startEvent");

  //TODO: Call cuBLAS SAXPY
  cublasSaxpy(cuBLAShandle, numElements, &alpha, xD,1,yD,1);
  cudaEventRecord ( endEvent, stream );
  checkError("Record CUDA event endEvent");
	
  //copy output to host
  cudaMemcpyAsync( yDeviceResultH, yD, numElements*sizeof(float), cudaMemcpyDeviceToHost, stream );
  checkError("Start copy device to host y");
	
  //Need to wait for upload to finish to avoid race condition
  cudaEventSynchronize ( startEvent );
 
  //TODO: move cudaStreamSynchronize after host calculation to allow overlap of device side saxpy and
  //	    host side saxpy (use time ./task2 to time runtime)
  //cudaStreamSynchronize( stream );
  //checkError("Synchronize CUDA stream");

  //Compute reference asynchronously on host
#pragma omp parallel for
  for (i=0; i<numElements; ++i)
    {
      yH[i] = alpha * xH[i] + yH[i];
    }

  //AFTER host calculation
  cudaStreamSynchronize( stream );
  checkError("Synchronize CUDA stream");
	
  // Compare results
  for (i=0; i<numElements; ++i)
    {
      absError = fabs ( yH[i] - yDeviceResultH[i] );
      sumAbsError += absError;
      if (absError > maxAbsError)
	maxAbsError = absError;
    }
	
  printf("maxAbsError = %f, sumAbsError = %f\n", maxAbsError, sumAbsError);
	
  cudaEventSynchronize ( endEvent );
  checkError("Synchronize CUDA event endEvent");
  cudaEventElapsedTime ( &runtime, startEvent, endEvent );
  checkError("Get Elapsed Time startEvent endEvent");
	
  if ( maxAbsError > 1E-6 )
    {
      printf("ERROR: Check correctnes of the code.\n");
    }
  else
    {
      printf("SAXPY Throughput %f (GB/s)\n", 3*numElements*sizeof(float)*oneSecondInMs / ( oneGigaByte * runtime ) );
    }
	
  // free host and device memory
  cudaFree( yD );
  cudaFree( xD );
  cudaFreeHost( yDeviceResultH );
  cudaFreeHost( yH );
  cudaFreeHost( xH );

  // destroy events, handle and stream
  cublasDestroy(cuBLAShandle);
  cudaStreamDestroy( stream );
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
