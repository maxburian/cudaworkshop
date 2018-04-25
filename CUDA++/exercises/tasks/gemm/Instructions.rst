.. role:: cpp(code)
   :language: c++

Instructions
============

In this exercise, we'll make our dgemm kernel work for arbitrary data types that support 
multiplication and addition by using templates.

The file ``gemm.cu`` contains a version of the matrix multiplication using shared 
memory. Use :cpp:`template <typename T>` and replace all occurences of :cpp:`double` in the 
functions by T.

Change your data type from  :cpp:`double` to :cpp:`float` in the main program. Does it get faster?

