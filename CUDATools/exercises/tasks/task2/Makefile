NVCC=nvcc
NVCC_FLAGS= -G -g -gencode arch=compute_35,code=sm_35 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60

.PHONY: all

all: gpu_print

gpu_print: gpu_print.cu Makefile
ifndef CUDA_PATH
 $(error Please load CUDA before building)
endif
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

.PHONY: clean

clean:
	rm gpu_print

.PHONY: run debug-nsight debug-cuda-gdb

run: all
	bsub -U cuda18_0 -I -R "rusage[ngpus_shared=1]" ./gpu_print

debug-nsight:
	module load openjdk && \
	bsub -U cuda18_0 -Is -R "rusage[ngpus_shared=1]" -tty -XF nsight
debug-cuda-gdb:
	bsub -U cuda18_0 -Is -R "rusage[ngpus_shared=1]" -tty cuda-gdb ./gpu_print
