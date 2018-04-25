NVCC = nvcc

NVFLAGS = --gpu-architecture=sm_60

.PHONY: all

all: jacobi

jacobi: jacobi.cu nvtx_macros.h Makefile
	$(NVCC) -o $@ -DNO_SYNC -DUSE_NVTX $(NVFLAGS) $< -lnvToolsExt 

.PHONY: clean

clean:
	rm -f jacobi

.PHONY: run profile memcheck

run: jacobi
	bsub -U cuda18_0 -I -R "rusage[ngpus_shared=1]" ./jacobi

profile: jacobi
	nvprof --unified-memory-profiling per-process-device ./jacobi

memcheck: jacobi
	cuda-memcheck ./jacobi
