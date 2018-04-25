/* Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
#include <cstdio>

#define CUDA_RT_CALL( call )                                                                        \
{                                                                                                   \
    cudaError_t cudaStatus = call;                                                                  \
    if ( cudaSuccess != cudaStatus )                                                                \
        fprintf(stderr, "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with %s (%d).\n",  \
                        #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus);     \
}

#ifdef USE_DOUBLE
    typedef double real;
    #define MPI_REAL_TYPE MPI_DOUBLE
#else
    typedef float real;
    #define MPI_REAL_TYPE MPI_FLOAT
#endif

__global__ void initialize_boundaries(
    real* __restrict__ const a_new,
    real* __restrict__ const a,
    const real pi,
    const int nx, const int ny )
{
    for (int iy = blockIdx.x * blockDim.x + threadIdx.x; 
         iy < ny; 
         iy += blockDim.x * gridDim.x) {
        const real y0 = sin( 2.0 * pi * iy / (ny-1) );
        a[     iy*nx + 0 ]         = y0;
        a[     iy*nx + (nx-1) ] = y0;
        a_new[ iy*nx + 0 ]         = y0;
        a_new[ iy*nx + (nx-1) ] = y0;
    }
}

void launch_initialize_boundaries(
    real* __restrict__ const a_new,
    real* __restrict__ const a,
    const real pi,
    const int nx, const int ny )
{
    initialize_boundaries<<<ny/128+1,128>>>( a_new, a, pi, nx, ny );
    CUDA_RT_CALL( cudaGetLastError() );
}

__global__ void jacobi_kernel(
          real* __restrict__ const a_new,
    const real* __restrict__ const a,
          real* __restrict__ const l2_norm,
    const int iy_start, const int iy_end,
    const int nx)
{
    for (int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start; 
         iy < iy_end; 
         iy += blockDim.y * gridDim.y) {
    for (int ix = blockIdx.x * blockDim.x + threadIdx.x + 1; 
         ix < (nx-1); 
         ix += blockDim.x * gridDim.x) {
        const real new_val = 0.25 * ( a[ iy * nx + ix + 1 ] + a[ iy * nx + ix - 1 ]
                                    + a[ (iy+1) * nx + ix ] + a[ (iy-1) * nx + ix ] );
        a_new[ iy * nx + ix ] = new_val;
        real residue = new_val - a[ iy * nx + ix ];
        atomicAdd( l2_norm, residue*residue );
    }}
}

void launch_jacobi_kernel(
          real* __restrict__ const a_new,
    const real* __restrict__ const a,
          real* __restrict__ const l2_norm,
    const int iy_start, const int iy_end,
    const int nx,
    cudaStream_t stream)
{
    dim3 dim_block(32,4,1);
    dim3 dim_grid( nx/dim_block.x+1, (iy_end-iy_start)/dim_block.y+1, 1 );
    jacobi_kernel<<<dim_grid,dim_block,0,stream>>>( a_new, a, l2_norm, iy_start, iy_end, nx );
    CUDA_RT_CALL( cudaGetLastError() );
}
