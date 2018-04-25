NVCC=nvcc
NVCC_FLAGS=-std=c++11 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60
NVCC_FLAGS+=-lineinfo 
NVCC_FLAGS+=--expt-extended-lambda 

.PHONY: run

all: run

for_each: for_each.cu Makefile
	$(NVCC) $(CFLAGS) $(NVCC_FLAGS) $< $(LIBS) -o $@


.PHONY: clean

clean:
	rm -f for_each

.PHONY: run profile memcheck

run: for_each
	bsub -I -R "rusage[ngpus_shared=1]" ./for_each

profile: for_each
	bsub -I -R "rusage[ngpus_shared=1]" nvprof --unified-memory-profiling per-process-device ./for_each

memcheck: for_each
	bsub -I -R "rusage[ngpus_shared=1]" cuda-memcheck ./for_each
	

