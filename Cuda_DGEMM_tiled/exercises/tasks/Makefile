NVCC = nvcc
OPT_FLAGS = -O3
OMP_FLAGS = -Xcompiler -fopenmp
NVFLAGS = --cudart=static -gencode arch=compute_60,code=compute_60 -std=c++11
CUBLASFLAGS = -lcublas


.phony: all
all: Cuda_DGEMM_tiled Cuda_DGEMM_tiled_streams Cuda_DGEMM_tiled_streams_multigpu Cuda_DGEMM_tiled_UVM Cuda_DGEMM_tiled_streams_UVM Cuda_DGEMM_cublasXt

Cuda_DGEMM_tiled: Cuda_DGEMM_tiled.cu
	$(NVCC) $(OPT_FLAGS) $(NVFLAGS) $(CUBLASFLAGS) $(OMP_FLAGS) -o Cuda_DGEMM_tiled Cuda_DGEMM_tiled.cu 

Cuda_DGEMM_tiled_streams: Cuda_DGEMM_tiled_streams.cu
	$(NVCC) $(NVFLAGS) $(OPT_FLAGS) $(CUBLASFLAGS) $(OMP_FLAGS) -o Cuda_DGEMM_tiled_streams Cuda_DGEMM_tiled_streams.cu 

Cuda_DGEMM_tiled_streams_multigpu: Cuda_DGEMM_tiled_streams_multigpu.cu
	$(NVCC) $(NVFLAGS) $(OPT_FLAGS) $(CUBLASFLAGS) $(OMP_FLAGS) -o Cuda_DGEMM_tiled_streams_multigpu Cuda_DGEMM_tiled_streams_multigpu.cu

Cuda_DGEMM_tiled_UVM: Cuda_DGEMM_tiled_UVM.cu
	$(NVCC) $(OPT_FLAGS) $(NVFLAGS) $(CUBLASFLAGS) $(OMP_FLAGS) -o Cuda_DGEMM_tiled_UVM Cuda_DGEMM_tiled_UVM.cu 

Cuda_DGEMM_tiled_streams_UVM: Cuda_DGEMM_tiled_streams_UVM.cu
	$(NVCC) $(NVFLAGS) $(OPT_FLAGS) $(CUBLASFLAGS) $(OMP_FLAGS) -o Cuda_DGEMM_tiled_streams_UVM Cuda_DGEMM_tiled_streams_UVM.cu 

Cuda_DGEMM_cublasXt: Cuda_DGEMM_cublasXt.cu
	$(NVCC) $(NVFLAGS) $(OPT_FLAGS) $(CUBLASFLAGS) $(OMP_FLAGS) -o Cuda_DGEMM_cublasXt Cuda_DGEMM_cublasXt.cu

.phony: clean
clean:
	rm -f Cuda_DGEMM_tiled
	rm -f Cuda_DGEMM_tiled_streams 
	rm -f Cuda_DGEMM_tiled_streams_multigpu
	rm -f Cuda_DGEMM_tiled_UVM
	rm -f Cuda_DGEMM_tiled_streams_UVM
	rm -f Cuda_DGEMM_cublasXt

.phony: run
run: run_task1 run_task2 run_task3 run_task1_UVM run_task2_UVM run_cublasXt

.phony: run_task1
run_task1: Cuda_DGEMM_tiled
	bsub -U cuda18_2 -I -R "rusage[ngpus_shared=1]" ./Cuda_DGEMM_tiled

.phony: run_task2
run_task2: Cuda_DGEMM_tiled_streams
	bsub -U cuda18_2 -I -R "rusage[ngpus_shared=1]" ./Cuda_DGEMM_tiled_streams

.phony: run_task3
run_task3: Cuda_DGEMM_tiled_streams_multigpu
	bsub -U cuda18_2 -I -R "rusage[ngpus_shared=1]" ./Cuda_DGEMM_tiled_streams_multigpu

.phony: run_task1_UVM
run_task1_UVM: Cuda_DGEMM_tiled_UVM
	bsub -U cuda18_2 -I -R "rusage[ngpus_shared=1]" ./Cuda_DGEMM_tiled_UVM

.phony: run_task2_UVM
run_task2_UVM: Cuda_DGEMM_tiled_streams_UVM
	bsub -U cuda18_2 -I -R "rusage[ngpus_shared=1]" ./Cuda_DGEMM_tiled_streams_UVM

.phony: run_cublasXt
run_cublasXt: Cuda_DGEMM_cublasXt
	bsub -U cuda18_2 -I -R "rusage[ngpus_shared=1]" ./Cuda_DGEMM_cublasXt 32768 4096 4
