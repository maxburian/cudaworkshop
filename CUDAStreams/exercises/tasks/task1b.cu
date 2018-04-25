#include <stdio.h>

void checkError (const char* action);

int main (int argc, char** argv)
{
    int retVal = 0;    			/* return value */
    int numElements = 1<<22; 	/* 4194304 */
    float runtime = 0.0f;
    double oneSecondInMs = 1000.0;
    int oneGigaByte = 1<<30;

    // Events to be used for timing
    cudaEvent_t startEvent;
    cudaEvent_t endEvent;
    
    // Streams to be created and used for parallel data transfers
    cudaStream_t uploadStream;
    cudaStream_t downloadStream;
    
    // Host and device pointers
    float* inputBufferH;
    float* outputBufferH;
    float* inputBufferD;
    float* outputBufferD;

    cudaSetDevice(0);
    checkError("Initialize CUDA device");

    // Set up events and streams
    cudaEventCreate( &startEvent );
    checkError("Create CUDA event startEvent");
    cudaEventCreate( &endEvent );
    checkError("Create CUDA event endEvent");

    // TODO : Create uploadStream and downloadStream CUDA streams ...
    
    // Allocate host and device memory
    cudaMallocHost( (void**) &inputBufferH, numElements*sizeof(float) );
    checkError("Allocate pinned host memory inputBufferH");
    cudaMallocHost( (void**) &outputBufferH, numElements*sizeof(float) );
    checkError("Allocate pinned host memory outputBufferH");
    cudaMalloc( (void**) &inputBufferD, numElements*sizeof(float) );
    checkError("Allocate device memory inputBufferD");
    cudaMalloc( (void**) &outputBufferD, numElements*sizeof(float) );
    checkError("Allocate device memory outputBufferD");

    /*event is recorded in default stream zero have timing of operations in both streams*/
    cudaEventRecord ( startEvent );
    checkError("Record CUDA event startEvent");
	
    //TODO: Start memcpys asynchronously in seperate streams to overlap the host to device with the device to host transfer
    cudaMemcpy( inputBufferD, inputBufferH, numElements*sizeof(float), cudaMemcpyHostToDevice);
    checkError("Copy host to device");
    cudaMemcpy( outputBufferH, outputBufferD, numElements*sizeof(float), cudaMemcpyDeviceToHost);
    checkError("Copy device to host");

    /*synchronize default stream this will also wait for work in uploadStream and downloadStream */
    cudaStreamSynchronize(0);
    cudaEventRecord ( endEvent );
    checkError("Record CUDA event endEvent");
    cudaEventSynchronize ( endEvent );
    checkError("Synchronize CUDA event endEvent");
    cudaEventElapsedTime ( &runtime, startEvent, endEvent );
    checkError("Get Elapsed Time startEvent endEvent");
	
    printf("Bidirectional Bandwidth %f (GB/s)\n", numElements*sizeof(float)*oneSecondInMs / ( oneGigaByte * runtime ) );
	
    // free host and device memory
    cudaFree( outputBufferD );
    cudaFree( inputBufferD );
    cudaFreeHost( outputBufferH );
    cudaFreeHost( inputBufferH );
    
    //TODO Destroy the user created streams  ...

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
