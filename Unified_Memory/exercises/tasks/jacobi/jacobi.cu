/* Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <cmath>
#include <iostream>
#include <cstdio>

#include "nvtx_macros.h"

#define CUDA_CALL( call )                                                                                          \
{                                                                                                                  \
  cudaError_t err = call;                                                                                          \
  if ( cudaSuccess != err)                                                                                         \
    fprintf(stderr, "CUDA error for %s in %d of %s : %s.\n", #call , __LINE__ , __FILE__ ,cudaGetErrorString(err));\
}

const float PI = 2.0f*std::asin(1.0f);

__global__ void jacobi_iteration(float       * __restrict__ const a_new,
                                 float const * __restrict__ const a,
                                 const int nx,
                                 const int ny,
                                 const float weight)
{
    for(int iy  = 1+blockIdx.y*blockDim.y+threadIdx.y;
            iy  < (ny-1);
            iy += gridDim.y*blockDim.y)
    {
        for(int ix  = 1+blockIdx.x*blockDim.x+threadIdx.x;
                ix  < (nx-1);
                ix += gridDim.x*blockDim.x)
        {
            
            const float a_new_val = 0.25f* ( a[(iy+0)*nx+(ix+1)]+a[(iy+0)*nx+(ix-1)]
                                           + a[(iy+1)*nx+(ix+0)]+a[(iy-1)*nx+(ix+0)]);
            a_new[iy*nx+ix] = weight*a_new_val+(1.0f-weight)*a[iy*nx+ix];
        }
    }
}

__global__ void apply_periodic_bc(float       * __restrict__ const a,
                                  const int nx,
                                  const int ny)
{
    for(int ix  = blockIdx.x*blockDim.x+threadIdx.x;
            ix  < nx;
            ix += gridDim.x*blockDim.x)
    {
        a[     0*nx+ix]=a[(ny-2)*nx+ix];
        a[(ny-1)*nx+ix]=a[     1*nx+ix];
    }
}

void init(float * __restrict__ const a,
          float * __restrict__ const a_new,
          const int nx,
          const int ny,
          float* __restrict__ const weights,
          const int n_weights)
{
    memset(a,     0, nx*ny*sizeof(float));
    memset(a_new, 0, nx*ny*sizeof(float));
    
    // set boundary conditions
    for (int iy = 0; iy < ny; ++iy)
    {
        const float y0      = std::sin( 2.0f * PI * iy / (ny-1));
        a    [iy*nx+0]      = y0;
        a    [iy*nx+(nx-1)] = y0;
        a_new[iy*nx+0]      = y0;
        a_new[iy*nx+(nx-1)] = y0;
    }
    for (int i = 0; i < n_weights; ++i)
    {
        weights[i] = 2.0f/3.0f;
    }
}

int main()
{
    int nx = 512;
    int ny = 512;
    int n_weights = 16;
    const int iter_max = 1000;
    
    float * a;
    float * a_new;
    float * weights;

    CUDA_CALL(cudaMallocManaged(&a,     nx*ny*sizeof(float)));
    CUDA_CALL(cudaMallocManaged(&a_new, nx*ny*sizeof(float)));
    CUDA_CALL(cudaMallocManaged(&weights, n_weights*sizeof(float)));

    init(a,a_new,nx,ny,weights,n_weights);
    
    cudaEvent_t start,stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaEventRecord(start));
    
    PUSH_RANGE("while loop",0)
    int iter = 0;

    while ( iter <= iter_max )
    {
        PUSH_RANGE("jacobi step",1)
        jacobi_iteration<<<dim3(nx/32,ny/4),dim3(32,4)>>>(a_new,a,nx,ny,weights[0]);
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());
        POP_RANGE
        
        std::swap(a,a_new);
        
        PUSH_RANGE("periodic boundary conditions",2)
        //Apply periodic boundary conditions
        // TODO: launch apply_periodic_bc<<<dim3(nx/128),dim3(128)>>> kernel to apply boundary conditions on the GPU instead of CPU for loop
        for (int ix = 0; ix < nx; ++ix)
        {
            a[     0*nx+ix]=a[(ny-2)*nx+ix];
            a[(ny-1)*nx+ix]=a[     1*nx+ix];
        }

        POP_RANGE
        
        if ( 0 == iter%100 )
        {
            std::cout<<iter<<std::endl;
        }
        iter++;
    }
    
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaDeviceSynchronize());
    POP_RANGE
    
    float runtime = 0.0f;
    CUDA_CALL(cudaEventElapsedTime(&runtime,start,stop));
    
    std::cout<<"Runtime "<<runtime/1000.0f<<" seconds."<<std::endl;

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(weights);
    cudaFree(a_new);
    cudaFree(a);
    cudaDeviceReset();
    
    return 0;
}

