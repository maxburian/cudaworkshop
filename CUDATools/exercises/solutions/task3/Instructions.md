# Task 3: Visual Profiler / `nvprof`

Have a look at the `scale_vector_um` from this morning's *Basics* session. Study it a bit with the Visual Profiler or `nvprof`.

* Compile with `make`
* Start an un-instrumented job with `make run`
* Profile the program:
    + **`nvprof`**: Call `make profile-nvprof` to see how to invoke `nvprof` in conjunction with the batch system. Adapt the command to generate a profile file which can be imported into a Visual Profiler session. Also, generate a metrics file and import it.
    + **Visual Profiler**: Launch an interactive session on JURON with 

        ```bash
        bsub -Is -R "rusage[ngpus_shared=1]" -XF -tty /bin/bash
        ```

        Then call Visual Profiler with `nvvp`. Attention: There are only limited amounts of compute nodes available! You might want to generate a profile with `nvprof` and import it into a Visual Profiler session running on JURON's login node.

        Use `make profile-nvvp` to automate the launch of the Visual Profiler. 
