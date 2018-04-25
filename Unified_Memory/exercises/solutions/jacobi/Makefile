NVCC=nvcc
NVCC_FLAGS=-gencode arch=compute_35,code=sm_35 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60
NVCC_FLAGS+=-lineinfo
CFLAGS=-DUSE_NVTX
LIBS=-lnvToolsExt

.PHONY: run

all: run

jacobi: jacobi.cu nvtx_macros.h Makefile
	$(NVCC) $(CFLAGS) $(NVCC_FLAGS) $< $(LIBS) -o $@

jacobi_nonsync: CFLAGS+=-DNO_SYNC
jacobi_nonsync: jacobi.cu nvtx_macros.h Makefile
	$(NVCC) $(CFLAGS) $(NVCC_FLAGS) $< $(LIBS) -o $@

.PHONY: clean

clean:
	rm -f jacobi

.PHONY: run profile memcheck

run: jacobi
	bsub -U cuda18_0 -I -R "rusage[ngpus_shared=1]" ./jacobi

profile: jacobi
	bsub -U cuda18_0 -I -R "rusage[ngpus_shared=1]" nvprof --unified-memory-profiling per-process-device ./jacobi

memcheck: jacobi
	bsub -U cuda18_0 -I -R "rusage[ngpus_shared=1]" cuda-memcheck ./jacobi
	

