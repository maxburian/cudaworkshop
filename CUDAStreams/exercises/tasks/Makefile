CUBLASFLAGS = -lcublas 
NVCC = nvcc
NVFLAGS = -O3 -Xcompiler -fopenmp --cudart=static -gencode arch=compute_60,code=compute_60 -std=c++11 

.phony: all
all: task1a task1b task2

task1a: task1a.cu
	$(NVCC) $(NVFLAGS) -o task1a task1a.cu

task1b: task1b.cu
	$(NVCC) $(NVFLAGS) -o task1b task1b.cu

task1a_um: task1a_um.cu
	$(NVCC) $(NVFLAGS) -o task1a_um task1a_um.cu

task1b_um: task1b_um.cu
	$(NVCC) $(NVFLAGS) -o task1b_um task1b_um.cu

task2: task2.cu
	$(NVCC) $(CUBLASFLAGS) $(NVFLAGS) -o task2 task2.cu 

.phony: clean
clean:
	rm -f task1a task1b task2

.phony: run
run: run_task1a run_task1b run_task2

.phony: run_task1a
run_task1a: task1a
	bsub -U cuda18_2 -I -R "rusage[ngpus_shared=1]" ./task1a

.phony: run_task1b
run_task1b: task1b
	bsub -U cuda18_2 -I -R "rusage[ngpus_shared=1]" ./task1b

.phony: run_task1a_um
run_task1a_um: task1a_um
	bsub -U cuda18_2 -I -R "rusage[ngpus_shared=1]" ./task1a_um

.phony: run_task1b_um
run_task1b_um: task1b_um
	bsub -U cuda18_2 -I -R "rusage[ngpus_shared=1]" ./task1b_um

.phony: run_task2
run_task2: task2
	bsub -U cuda18_2 -I -R "rusage[ngpus_shared=1]" ./task2

