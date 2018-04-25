CC = xlc
NVCC = nvcc
MPICC = mpicc

MKLFLAGS = -lesslsmp -lessl -lpthread -lm -ldl
CUBLASFLAGS = -I${CUDA_ROOT}/include -L${CUDA_ROOT}/lib64 -lcublas -lcudart
CFLAGS = -g -O3
NVFLAGS = --cudart=static -gencode arch=compute_30,code=compute_30 -gencode 
arch=compute_60,code=compute_60 -std=c++11

.phony: all

all: dgemm_um
	
dgemm_um: dgemm_um.cpp
	$(CC) -o dgemm_um dgemm_um.cpp $(MKLFLAGS) $(CUBLASFLAGS) $(CFLAGS)
	
.phony: clean

clean:
	rm -f *.o dgemm_um

.phony: run

run: all
	bsub -U cuda18_0 -I -R "rusage[ngpus_shared=1]" ./dgemm_um 100
