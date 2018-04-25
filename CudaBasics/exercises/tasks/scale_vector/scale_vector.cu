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
    float* a = (float*) malloc(m * sizeof(float));
    float* c = (float*) malloc(m * sizeof(float));
    for (int i = 0; i < m; ++i) a[i] = 1.0;
    float* a_gpu;
    float* c_gpu;
    // TODO: Allocate a_gpu and c_gpu on the device
    cudaMalloc(&a_gpu, m*sizeof(float));
    cudaMalloc(&c_gpu, m*sizeof(float));

    // TODO: Copy a into a_gpu
    cudaMemcpy(a_gpu, a, m*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(c_gpu, c, m*sizeof(float), cudaMemcpyDeviceToHost);

    // TODO: Define a 1d thread block of length 256
    dim3 blockDim(256);

    dim3 gridDim((m % 256) ? m / blockDim.x : m / blockDim.x + 1);
//     printf("gridDim(%d,%d)", gridDim.x, gridDim.y);
    
    // TODO: Call the kernel
    scale<<<gridDim, blockDim>>>(alpha, a_gpu, c_gpu, m);
    
    // TODO: Copy c_gpu into c
    cudaMemcpy(c, c_gpu, m*sizeo

    // TODO: Free memory on device


    for (int i = 0; i < m; ++i){
        if (abs(c[i] - alpha * a[i]) > tolerance){
            printf("Failed! Element %d: c[%d] != %f a[%d] (%f != %f * %f)\n", i, i,alpha,i,c[i], alpha,a[i]);
            return 1;
        }
    }
    printf("Passed!\n");
    return 0;
}
