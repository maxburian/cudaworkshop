CC = xlc
NVCC = nvcc
MPICC = mpicc

MKLFLAGS = -lesslsmp -lessl -lpthread -lm -ldl
CUBLASFLAGS = -I${CUDA_ROOT}/include -L${CUDA_ROOT}/lib64 -lcublas -lcudart
CFLAGS = -g -O3
CXXFLAGS = -std=c++11
NVFLAGS = --cudart=static -gencode arch=compute_30,code=compute_30 -gencode arch=compute_60,code=compute_60 -std=c++11

.phony: all

all: dgemm ddot mandelbrot mandelbrot_cpu
	
dgemm: dgemm_essl.cpp
	$(CC) -o dgemm dgemm_essl.cpp $(MKLFLAGS) $(CUBLASFLAGS) $(CFLAGS)
	
ddot: ddot.cpp
	$(CC) -o ddot ddot_essl.cpp $(MKLFLAGS) $(CUBLASFLAGS) $(CFLAGS)
	
mandelbrot: mandelbrot.cu
	$(NVCC) -o mandelbrot mandelbrot.cu $(NVFLAGS)

mandelbrot_cpu: mandelbrot.cpp
	$(CXX) -o mandelbrot_cpu mandelbrot.cpp $(CXXFLAGS) $(CFLAGS) -fopenmp

.phony: clean veryclean

clean:
	rm -f *.o dgemm ddot mandelbrot mandelbrot_cpu

veryclean: clean
	rm -f *.pdf

.PHONY: run gen_montage display

run: all
	./bench_all.sh

gen_montage: all
	montage dgemm_bench.pdf nbody_bench.pdf ddot_bench.pdf mandelbrot_bench.pdf -tile 2x2 -geometry +0+0 benchmarks.pdf
display: all
	montage dgemm_bench.pdf nbody_bench.pdf ddot_bench.pdf mandelbrot_bench.pdf -tile 2x2 -geometry +0+0 - | display -resize 1280x1024 -
