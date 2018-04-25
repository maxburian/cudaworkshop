#define SOLUTION = 1
#include <iostream>
#include <chrono>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>

constexpr int gpuThreshold=10000;

void saxpy(float *x, float *y, float a, int N) {
    auto r = thrust::counting_iterator<int>(0);
// TODO: Write a lambda function that implements a * x + y
    auto lamda = ...

    if(N > gpuThreshold)
// TODO: Use thrust::for_each to call the lambda function for each element.
      
    else
      thrust::for_each(thrust::host, r, r+N, lambda);
    
    cudaDeviceSynchronize();
}

int main(int argc, char** argv){
    size_t N = (argc > 1) ? std::stoi(argv[1]) : 1000;
    float* x;
    float* y;
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));
    for(int i = 0; i < N; ++i){
        x[i] = i;
        y[i] = 1;
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    saxpy(x, y, 3.1415, N);
    auto end_time = std::chrono::high_resolution_clock::now();
    int errorCount = 0;
    for(int i = 0; i < N; ++i){
        if (abs(y[i] - (3.1415 * i + 1)) > (i * 1.0e-5)){
            std::cout << "There's an error in element " << i << "! ";
            std::cout << "y[" << i << "] = " << y[i] << " not " << (3.1415 * i + 1) << ".\n";
            errorCount++;
        }
    }
    if (errorCount == 0){
        std::cout << "saxpy with " << N << " elements took "
        << std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() * 
            1e-6 << " s" << std::endl;
    }
    cudaFree(x);
    cudaFree(y);
}
