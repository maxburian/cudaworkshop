# Task 1: Memory Hints

Again, look at `scale_vector_um`. Augment the code with information about data locality and movement.

* Compile with `make`
* Launch a batch job with `make run`
* Profile the program
    - `nvprof`: Use `make profile-nvprof` to profile to program manually with `nvprof`. Remember, you can export a profile with `-o filename` and import it to the Visual Profiler
    - Visual Profiler: Use `make profile-nvvp` to launch an instance of the Visual Profiler in which you profile your application
