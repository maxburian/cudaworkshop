NVCC=nvcc
NVCC_FLAGS=-std=c++11 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_35,code=compute_35 \
-gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60
NVCC_FLAGS+=-lineinfo

.PHONY: run

all: run

gemm: gemm.cu Makefile
	$(NVCC) $(CFLAGS) $(NVCC_FLAGS) $< $(LIBS) -o $@


.PHONY: clean

clean:
	rm -f gemm

.PHONY: run profile memcheck

run: gemm
	bsub -I -R "rusage[ngpus_shared=1]" ./gemm

profile: gemm
	bsub -I -R "rusage[ngpus_shared=1]" nvprof --unified-memory-profiling per-process-device ./gemm

memcheck: gemm
	bsub -I -R "rusage[ngpus_shared=1]" cuda-memcheck ./gemm
	

