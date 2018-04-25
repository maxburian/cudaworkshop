NVCC=nvcc
NVCC_FLAGS= -G -g -gencode arch=compute_35,code=sm_35 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60

.PHONY: all

all: set_vector

set_vector: set_vector.cu Makefile check-env
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

.PHONY: clean check-env

clean:
	rm set_vector

.PHONY: run memcheck

run: all
	bsub -U cuda18_0 -I -R "rusage[ngpus_shared=1]" ./set_vector

memcheck: all
	bsub -U cuda18_0 -I -R "rusage[ngpus_shared=1]" cuda-memcheck ./set_vector

check-env:
ifndef CUDA_PATH
 $(error Please load CUDA before building)
endif
