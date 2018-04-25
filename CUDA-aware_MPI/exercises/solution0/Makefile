# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
NP ?= 2
NVCC=nvcc
MPICXX=mpicxx
GENCODE_SM30	:= -gencode arch=compute_30,code=sm_30
GENCODE_SM35	:= -gencode arch=compute_35,code=sm_35
GENCODE_SM37	:= -gencode arch=compute_37,code=sm_37
GENCODE_SM50	:= -gencode arch=compute_50,code=sm_50
GENCODE_SM52	:= -gencode arch=compute_52,code=sm_52
GENCODE_SM60	:= -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60
GENCODE_FLAGS	:= $(GENCODE_SM60)
NVCC_FLAGS = -lineinfo $(GENCODE_FLAGS) -std=c++11
MPICXX_FLAGS = -DUSE_NVTX -I$(CUDA_HOME)/include -std=c++11
LD_FLAGS = -L$(CUDA_HOME)/lib64 -lcudart -lnvToolsExt
jacobi: Makefile jacobi.cpp jacobi_kernels.o
	$(MPICXX) $(MPICXX_FLAGS) jacobi.cpp jacobi_kernels.o $(LD_FLAGS) -o jacobi

jacobi_kernels.o: Makefile jacobi_kernels.cu
	$(NVCC) $(NVCC_FLAGS) jacobi_kernels.cu -c

.PHONY.: clean
clean:
	rm -f jacobi jacobi_kernels.o *.nvprof

memcheck: jacobi
	bsub -U cuda18_1 -x -n $(NP) -R "rusage[ngpus_shared=20]" -tty -Is mpirun --mca btl self,smcuda --mca oob ^ud --npersocket 2 -bind-to core -np $(NP) cuda-memcheck --log-file jacobi.%q{OMPI_COMM_WORLD_RANK}.memcheck.log ./jacobi

run: jacobi
	bsub -U cuda18_1 -x -n $(NP) -R "rusage[ngpus_shared=20]" -tty -Is mpirun --mca btl self,smcuda --mca oob ^ud --npersocket 2 -bind-to core -np $(NP) ./jacobi

profile: jacobi
	bsub -U cuda18_1 -x -n $(NP) -R "rusage[ngpus_shared=20]" -tty -Is mpirun --mca btl self,smcuda --mca oob ^ud --npersocket 2 -bind-to core -np $(NP) nvprof -o jacobi.%q{OMPI_COMM_WORLD_RANK}.nvprof --process-name "rank %q{OMPI_COMM_WORLD_RANK}" --context-name "rank %q{OMPI_COMM_WORLD_RANK}"  ./jacobi -niter 10
