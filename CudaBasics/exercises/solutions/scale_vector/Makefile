NVCC = nvcc

NVFLAGS = --gpu-architecture=sm_60

.PHONY: all

all: scale_vector_um
	
scale_vector_um: scale_vector_um.cu Makefile
	$(NVCC) -o $@ $(NVFLAGS) $<	

scale_vector: scale_vector.cu Makefile
	$(NVCC) -o $@ $(NVFLAGS) $<
	
.PHONY: clean

clean:
	rm -f scale_vector scale_vector_um

.PHONY: run

run: all
	bsub -U cuda18_0 -I -R "rusage[ngpus_shared=1]" ./scale_vector_um
