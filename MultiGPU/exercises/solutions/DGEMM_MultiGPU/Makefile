all:
	nvcc -O3 -arch=sm_20 -o dgemm_multigpu dgemm_multigpu.cu -lcudart
	nvcc -O3 -arch=sm_20 -o dgemm_gpu_simple dgemm_gpu_simple.cu -lcudart

clean:
	rm -rf dgemm_multigpu dgemm_gpu_simple
