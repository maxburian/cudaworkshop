/** This example uses cudaManagedAllocation (available since CUDA 6.0) to allocate memory and manage
 *  the memory transfer to and from the device.
 */

#include <stdio.h>

__global__ void scale(float alpha, float* a, float* c, int m){
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < m){
        c[i] = alpha * a[i];
    }
}

int main(int argc, char** argv){

    int device = 0;
    if(argc > 1) {
        device = atoi(argv[1]);
//         printf("Using device %d\n", device);
    }
    
    cudaSetDevice(device);
    
    
    int m = 2048;
    float alpha = 2.0;
    float tolerance = 1e-3f;
    float* a;
    float* c;
    // TODO: Use cudaMallocManaged to allocate space for a and c
    cudaMallocManaged(&a, m * sizeof(float));
    cudaMallocManaged(&c, m * sizeof(float));
    for (int i = 0; i < m; ++i) a[i] = 1.0;
    // TODO: Define a 1d thread block of length 256
    dim3 blockDim(256);

    dim3 gridDim((m % 256) ? m / blockDim.x : m / blockDim.x + 1);
//     printf("gridDim(%d,%d)", gridDim.x, gridDim.y);
    
    // TODO: Call the kernel
    scale<<<gridDim, blockDim>>>(alpha, a, c, m);

    // Note: Don't forget to synchronize before you want to access the data on the host since CUDA
    //       calls are asynchronous.
    cudaDeviceSynchronize();
   
    for (int i = 0; i < m; ++i){
        if (abs(c[i] - alpha * a[i]) > tolerance){
            printf("Failed! Element %d: c[%d] != %f a[%d] (%f != %f * %f)\n", i, i,alpha,i,c[i], alpha,a[i]);
            return 1;
        }
    }
    printf("Passed!\n");
    // TODO: Free memory
    cudaFree(a);
    cudaFree(c);
    return 0;
}
