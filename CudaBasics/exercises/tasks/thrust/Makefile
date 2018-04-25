NVCC = nvcc

NVFLAGS = --gpu-architecture=sm_60 -std=c++11 -lcurand

.PHONY: all

all: ThrustSort
	
ThrustSort: ThrustSort.cu Makefile
	$(NVCC) -o $@ $(NVFLAGS) $<
	
.PHONY: clean

clean:
	rm -f ThrustSort

.PHONY: run

run: all
	bsub -U cuda18_0 -Is -R "rusage[ngpus_shared=1]" ./ThrustSort
