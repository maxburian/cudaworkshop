NVCC=nvcc
NVCC_FLAGS= -G -g -gencode arch=compute_35,code=sm_35 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60

.PHONY: all

all: scale_vector_um
	
scale_vector_um: scale_vector_um.cu Makefile
ifndef CUDA_PATH
 $(error Please load CUDA before building)
endif
	$(NVCC) -o $@ $(NVFLAGS) $<	

scale_vector: scale_vector.cu Makefile
ifndef CUDA_PATH
 $(error Please load CUDA before building)
endif
	$(NVCC) -o $@ $(NVFLAGS) $<
	
.PHONY: clean

clean:
	rm -f scale_vector scale_vector_um

.PHONY: run profile

run: all
	bsub -U cuda18_0 -Is -R "rusage[ngpus_shared=1]" ./scale_vector_um

profile-nvvp: all
	module load openjdk && \
	bsub -U cuda18_0 -Is -R "rusage[ngpus_shared=1]" -XF nvvp
profile-nvprof: all
	bsub -U cuda18_0 -Is -R "rusage[ngpus_shared=1]" nvprof ./scale_vector_um
