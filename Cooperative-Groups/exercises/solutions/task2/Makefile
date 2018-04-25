NVCC=nvcc
NVCC_FLAGS= -lineinfo -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60

.PHONY: all

all: cg-reduce

cg-reduce: cg-reduce.cu Makefile
ifndef CUDA_PATH
 $(error Please load CUDA before building)
endif
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

.PHONY: clean

clean:
	rm cg-reduce

.PHONY: run memcheck

run: all
	bsub -U cuda18_2 -I -R "rusage[ngpus_shared=1]" ./cg-reduce
