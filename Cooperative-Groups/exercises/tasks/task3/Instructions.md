# Cooperative Groups: Task 3

With the tiled partitions from Task 2 we simplify the code to use warp-level intrinsic operations.

In comparison to Task 2, the base code here is templated with the tile size as the template parameter.

Go through the TODOs to remove all references to shared memory in the calling kernel and reduce the `maxFunction()` to a for loop over the group intrinsic `shfl_down()`. We can directly access the data elements stored in different lanes of the warp.

The definition of `shfl_down()` is 

```cpp
T shfl_down(T value, unsigned int delta)
```

As all warp-level operations (with CUDA 9), the function synchronizes its parent group. Explicit `sync()` calls are not needed.


See also [the CUDA documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions).
