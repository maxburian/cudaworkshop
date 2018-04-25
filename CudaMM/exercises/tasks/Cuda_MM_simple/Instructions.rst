Cuda MM Exercise - Simple GPU implementation
--------------------------------------------

*files:* Cuda_MM_simple.cu, CPU_MM_simple

a) Implement a matrix multiplication with Cuda. Compile and run
   the program using the existing Makefile:

   --> make Cuda_MM_simple
   --> make runGPU

   Run your program with different matrix sizes (can be specified as
   command line parameter, see Makefile on how to launch program manually)
   and take a look at the resulting performance.
   Use an interactive session to run your program with the visual
   profiler (nvvp). 

b) Replace the register variable ``Cvalue`` with direct access to the
   global memoray array (``c[...]``). Do not forget to initialize
   ``c[..]`` with zero. How does this change affect the execution time ? 

c) For comparison compile and run the CPU version of the matrix
   multiplication:

   --> make CPU_MM_simple
   --> make runCPU

d) If you have time left:
   Move to the solutions folder and take a look at the GPU versions
   using explicit memory management (``cudaMalloc`` and
   ``cudaMemcpy``). Run the correspondent executables:

   --> make Cuda_MM_simple_memcpy
   --> make runGPUmemcpy
   --> make Cuda_MM_simple_memcpy
   --> make runGPUmemcpyNoTransfers
  
