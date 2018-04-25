NVCC=nvcc

CFLAGS=-Xcompiler -std=c99
OMPCFLAGS=-DOMP_MEMLOCALTIY -Xcompiler -fopenmp
NVCCFLAGS=-O3 -arch=sm_37 -Xcompiler -march=native -lineinfo

OMPLDFLAGS=-Xcompiler -fopenmp

BINDIR=.
JACOBI_CUDA=$(BINDIR)/jacobi_cuda
BINARIES=$(JACOBI_CUDA)

all: $(BINARIES)

jacobi_cuda_kernel.o: jacobi_cuda_kernel.cu Makefile
	$(NVCC) $(NVCCFLAGS) -c jacobi_cuda_kernel.cu -o jacobi_cuda_kernel.o
	
jacobi_cuda.o: jacobi_cuda.c Makefile
	$(NVCC) $(CFLAGS) $(OMPCFLAGS) $(NVCCFLAGS) -c jacobi_cuda.c -o jacobi_cuda.o
	
$(JACOBI_CUDA): jacobi_cuda.o jacobi_cuda_kernel.o Makefile
	$(NVCC) $(NVCCFLAGS) $(OMPLDFLAGS) jacobi_cuda.o jacobi_cuda_kernel.o -o $(JACOBI_CUDA)
	
run: $(JACOBI_CUDA)
	$(JACOBI_CUDA) 2

profile: $(JACOBI_CUDA)
	nvprof -o jacobi_cuda.nvvp $(JACOBI_CUDA) 2

clean:
	rm -f *.o $(BINARIES) jacobi_cuda.nvvp
