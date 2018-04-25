/*
 * Copyright 2012 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <cuda_runtime.h>

/**
 * @brief Does one Jacobi iteration on u_d writing the results to
 *        unew_d on all interior points of the domain.
 *
 * The Jacobi iteration solves the poission equation with diriclet
 * boundary conditions and a zero right hand side executes asynchronously.
 *
 * @param[in] u_d            pointer to device memory holding the
 *                           solution of the last iteration including
 *                           boundary.
 * @param[out] unew_d        pointer to device memory were the updates
 *                           solution should be written
 * @param[in] n              number of points in y direction
 * @param[in] m              number of points in x direction
 * @param[in,out] residue_d  pointer to a single float value in device
 *               memory, needed a a temporary storage to
 *               calculate the max norm of the residue.
 */
void launch_jacobi_kernel(const float* const u_d, float* const unew_d,
                           const int n, const int m, float* const residue_d);

/**
 * @brief Copies all inner points from unew_d to u_d, executes
 *        asynchronously.
 *
 * @param[out] u_d    pointer to device memory holding the solution of
 *            the last iteration including boundary which
 *            should be updated with unew_d
 * @param[in] unew_d  pointer to device memory were the updated
 *                    solution is saved
 * @param[in] n       number of points in y direction
 * @param[in] m       number of points in x direction
 */
void launch_copy_kernel(float* const u_d, const float* const unew_d,
                        const int n, const int m);

void checkCUDAError(const char* action);
#define CUDA_CALL( call )       \
{                               \
    call;                       \
    checkCUDAError( #call );    \
}

int handle_command_line_arguments(int argc, char** argv);

int init(int argc, char** argv);

void finalize();

void start_timer();
void stop_timer();

void jacobi();

int n, m;
int n_global;

int rank = 0;
int size = 1;

int iter = 0;
int iter_max = 1000;

double starttime;
double runtime;

const float pi = 3.1415926535897932384626f;
const float tol = 1.0e-5f;
float residue = 1.0f;

float* u;
float* unew;
float* y0;

#define MAX_GPUS 16

float* u_d[MAX_GPUS];
float* unew_d[MAX_GPUS];
float* residue_d[MAX_GPUS];

/********************************/
/****         MAIN            ***/
/********************************/
int main(int argc, char** argv)
{
    if ( init(argc, argv) )
    {
        return -1;
    }

    start_timer();

    // Main calculation
    jacobi();

    stop_timer();

    finalize();
}

/********************************/
/****        JACOBI           ***/
/********************************/
void jacobi()
{
    while (residue > tol && iter < iter_max)
    {
        for ( int rank=0; rank < size; ++rank )
        {
            //TODO: launch jacobi kernel on the correct GPU using cudaSetDevice
            launch_jacobi_kernel(u_d[rank], unew_d[rank], n, m, residue_d[rank]);
        }
        
        residue = 0.0f;
        for ( int rank=0; rank < size; ++rank )
        {
            float res = 0.0f;
            CUDA_CALL( cudaMemcpy( &res, residue_d[rank], sizeof(float), cudaMemcpyDeviceToHost ) );
            residue = res > residue ? res : residue;
        }
        
        if ( size == 2 )
        {
            //TODO: Use cudaMemcpy( dst, src, size, cudaMemcpyDeviceToDevice ) to update halos
            //1. Copy last modified row of GPU 0 to upper boundary of GPU 1:
            //CUDA_CALL( cudaMemcpy ( u_d[1]+0*m+1, unew_d[0]+(n-2)*m+1, (m-2)*sizeof(float), cudaMemcpyDeviceToDevice ) );
            //2. Copy first modified row of GPU 1 to lower boundary of GPU 0:
            //CUDA_CALL( cudaMemcpy ( /*TODO*/, /*TODO*/, (m-2)*sizeof(float), cudaMemcpyDeviceToDevice ) );
        }
        
        for ( int rank=0; rank < size; ++rank )
        {
            //TODO: launch jacobi kernel on the correct GPU using cudaSetDevice
            launch_copy_kernel(u_d[rank], unew_d[rank], n, m);
        }

        if (iter % 100 == 0)
            printf("%5d, %0.6f\n", iter, residue);

        iter++;
    }
    for ( int rank=0; rank < size; ++rank )
    {
        CUDA_CALL( cudaMemcpy( u + rank*m*(n-2), u_d[rank], m*n*sizeof(float), cudaMemcpyDeviceToHost ));
    }
}

/********************************/
/**** Initialization routines ***/
/********************************/

int init(int argc, char** argv)
{
    if (handle_command_line_arguments(argc, argv)) {
        return -1;
    }
    
    if ( size != 1 && size != 2 )
    {
        printf("Error: %s can only run with 1 or 2 GPUs!\n",argv[0]);
        return -1;
    }

    u = (float*) malloc(n_global * m * sizeof(float));
    unew = (float*) malloc(n_global * m * sizeof(float));
    y0 = (float*) malloc(n_global * sizeof(float));

#ifdef OMP_MEMLOCALTIY
#pragma omp parallel for
    for( int j = 0; j < n_global; j++)
    {
        for( int i = 0; i < m; i++ )
        {
            unew[j *m+ i] = 0.0f;
            u[j *m+ i] = 0.0f;
        }
    }
#else
    memset(u, 0, n_global * m * sizeof(float));
    memset(unew, 0, n_global * m * sizeof(float));
#endif //OMP_MEMLOCALTIY

    // set boundary conditions
#pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
        u[0 * m + i] = 0.f;
        u[(n_global - 1) * m + i] = 0.f;
    }

    for (int j = 0; j < n_global; j++)
    {
        y0[j] = sinf(pi * j / (n_global - 1));
        u[j * m + 0] = y0[j];
        u[j * m + (m - 1)] = y0[j] * expf(-pi);
    }

#pragma omp parallel for
    for (int i = 1; i < m; i++)
    {
        unew[0 * m + i] = 0.f;
        unew[(n_global - 1) * m + i] = 0.f;
    }
#pragma omp parallel for
    for (int j = 1; j < n_global; j++)
    {
        unew[j * m + 0] = y0[j];
        unew[j * m + (m - 1)] = y0[j] * expf(-pi);
    }

    for ( int rank=0; rank < size; ++rank )
    {
        //TODO: Use cudaSetDevice to allocated memory and initialize memory on the correct GPU
        CUDA_CALL( cudaMalloc( (void**)&(u_d[rank]), n*m * sizeof(float) ));
        CUDA_CALL( cudaMalloc( (void**)&(unew_d[rank]), n*m * sizeof(float) ));
        CUDA_CALL( cudaMalloc( (void**)&(residue_d[rank]), sizeof(float) ));
        CUDA_CALL( cudaMemcpy( u_d[rank], u + rank*m*(n-2), m*n*sizeof(float), cudaMemcpyHostToDevice ));
        CUDA_CALL( cudaMemcpy( unew_d[rank], unew + rank*m*(n-2), m*n*sizeof(float), cudaMemcpyHostToDevice ));
    }
    
    if ( size == 2 )
    {
        int canAccessPeer = 0;
        //TODO: Check if peer access is possible cudaDeviceCanAccessPeer ( &canAccessPeer, int  device, int  peerDevice )
        if ( canAccessPeer )
        {
            //TODO: Enable peer access using cudaSetDevice and cudaDeviceEnablePeerAccess( int  peerDevice, 0 )
        }
    }
    return 0;
}

int handle_command_line_arguments(int argc, char** argv)
{
    if (argc > 4 || argc < 2)
    {
        printf("usage: %s num_gpus [n] [m]\n", argv[0]);
        return -1;
    }
    
    int num_gpus=0;
    CUDA_CALL( cudaGetDeviceCount ( &num_gpus ) );
    size = atoi(argv[1]);
    if ( size > num_gpus || size <= 0 || size > 2 )
    {
        printf("Error: %i GPUs requested. %i are available and 1 or 2 are supported!\n",size,num_gpus);
        return -1;
    }

    n = 4096;
    if (argc >= 3)
    {
        n = atoi(argv[2]);
        if (n <= 0)
        {
            printf("Error: The number of rows (n=%i) needs to positive!\n",n);
            return -1;
        }
    }

    if (size == 2 && n % 2 != 0)
    {
        printf( "Error: The number of rows (n=%i) needs to be divisible by 2 if two GPUs are used!\n",n);
        return -1;
    }
    m = n;
    if (argc >= 4)
    {
        m = atoi(argv[3]);
        if (m <= 0)
        {
            printf( "Error: The number of columns (m=%i) needs to positive!\n", m);
            return -1;
        }
    }

    n_global = n;

    if (size == 2)
    {
        //Do a domain decomposition and add one row for halo cells
        n = n / 2 + 1;
    }


    printf("Jacobi relaxation Calculation: %d x %d mesh with:\n", n_global, m );
    for ( int rank=0; rank < size; ++rank )
    {
        struct cudaDeviceProp devProp;
        CUDA_CALL( cudaGetDeviceProperties( &devProp, rank ));
        printf("  GPU %d (%s) processing %d rows.\n",
                rank, devProp.name,n);
    }
    return 0;
}

/********************************/
/****  Finalization routines  ***/
/********************************/

void finalize()
{
    for ( int rank=0; rank < size; ++rank )
    {
        CUDA_CALL( cudaFree( residue_d[rank] ));
        CUDA_CALL( cudaFree( unew_d[rank] ));
        CUDA_CALL( cudaFree( u_d[rank] ));
    }

    free(y0);
    free(unew);
    free(u);
}

/********************************/
/****    Timing functions     ***/
/********************************/
void start_timer()
{
    starttime = omp_get_wtime();
}

void stop_timer()
{
    runtime = omp_get_wtime() - starttime;
    printf(" total: %f s\n", runtime);
}
