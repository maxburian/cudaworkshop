CC = gcc
NVCC = nvcc
CFLAGS = -g -O3 -fopenmp
NVFLAGS = -O3 --cudart=static -gencode arch=compute_60,code=compute_60 -std=c++11

.phony: all

all: CPU_MM_simple Cuda_MM_simple Cuda_MM_simple_memcpy Cuda_MM_simple_memcpy_no_transfer_times

CPU_MM_simple: CPU_MM_simple.c
	$(CC) -o CPU_MM_simple CPU_MM_simple.c $(CFLAGS)

Cuda_MM_simple: Cuda_MM_simple.cu
	$(NVCC) -o Cuda_MM_simple Cuda_MM_simple.cu $(NVFLAGS) 

Cuda_MM_simple_memcpy: Cuda_MM_simple_memcpy.cu
	$(NVCC) -o Cuda_MM_simple_memcpy Cuda_MM_simple_memcpy.cu $(NVFLAGS)

Cuda_MM_simple_memcpy_no_transfer_times: Cuda_MM_simple_memcpy_no_transfer_times.cu
	$(NVCC) -o Cuda_MM_simple_memcpy_no_transfer_times Cuda_MM_simple_memcpy_no_transfer_times.cu $(NVFLAGS)

.phony: clean

clean:
	rm -f *.o CPU_MM_simple Cuda_MM_simple Cuda_MM_simple_memcpy Cuda_MM_simple_memcpy_no_transfer_times

.phony: run

run: all runCPU runGPU runGPUmemcpy runGPUmemcpyNoTransfers

.phony: runCPU

runCPU: CPU_MM_simple
	bsub -I ./CPU_MM_simple

.phony: runGPU

runGPU: Cuda_MM_simple
	bsub -I -R "rusage[ngpus_shared=1]" ./Cuda_MM_simple

.phony: profileGPU

profileGPU: Cuda_MM_simple
	bsub -I -R "rusage[ngpus_shared=1]" nvprof --analysis-metrics -f -o f2 ./Cuda_MM_simple

.phony: runGPUmemcpy

runGPUmemcpy: Cuda_MM_simple_memcpy
	bsub -I -R "rusage[ngpus_shared=1]" ./Cuda_MM_simple_memcpy

.phony: runGPUmemcpyNoTransfers

runGPUmemcpyNoTransfers: Cuda_MM_simple_memcpy_no_transfer_times
	bsub -I -R "rusage[ngpus_shared=1]" ./Cuda_MM_simple_memcpy_no_transfer_times


