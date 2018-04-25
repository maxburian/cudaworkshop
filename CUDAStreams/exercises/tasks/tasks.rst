Cuda Streams Exercise - Use Cuda Streams, Events and Asynchronous Memcopy
-------------------------------------------------------------------------

*files:* task1a.cu, task1b.cu, task2.cu

task1
*****

a) Follow the TODOs in CUDAStreams/exercises/tasks/task1a.cu and
   allocate host buffers in pinned memory. Compile and run using

   --> make task1a
   --> make run_task1a

   Use an interactive session to run your program with visual profiler
   (nvvp).


b) Follow the TODOs in CUDAStreams/exercises/tasks/task1b.cu and
   use different streams for upload and download. Issue Host to Device 
   and Device to Host Transfer asynchronously in the two new streams.
   Compile and run using 

   --> make task1b
   --> make run_task1b

   Use an interactive session to run your program with visual profiler
   (nvvp).


task2
*****

Follow the TODOs in CUDAStreams/exercises/tasks/task2.cu to set CUBLAS
execution stream, call CUBLAS SAXPY and to fix position of
cudaStreamSynchronize. Compile and run using

--> make task2
--> make run_task2

Use an interactive session to run your program with visual profiler
(nvvp).
