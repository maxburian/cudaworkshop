NVCC = nvcc
NVFLAGS = -O3 --cudart=static -gencode arch=compute_60,code=compute_60 -std=c++11

.phony: all

all: Cuda_MM_shared Cuda_MM_shared_memcpy

Cuda_MM_shared: Cuda_MM_shared.cu
	$(NVCC) -o Cuda_MM_shared Cuda_MM_shared.cu $(NVFLAGS) 

Cuda_MM_shared_memcpy: Cuda_MM_shared_memcpy.cu
	$(NVCC) -o Cuda_MM_shared_memcpy Cuda_MM_shared_memcpy.cu $(NVFLAGS)

.phony: clean

clean:
	rm -f *.o Cuda_MM_shared Cuda_MM_shared_memcpy

.phony: run

run: all runGPU runGPUMemcpy

.phony: runGPU

runGPU: Cuda_MM_shared
	bsub -I -R "rusage[ngpus_shared=1]" -U cuda18_2 ./Cuda_MM_shared

.phony: runGPUMemcpy

runGPUMemcpy: Cuda_MM_shared_memcpy
	bsub -I -R "rusage[ngpus_shared=1]" -U cuda18_2  ./Cuda_MM_shared_memcpy
