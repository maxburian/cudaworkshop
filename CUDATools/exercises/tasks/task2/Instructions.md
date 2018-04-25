# Task 2: Nsight Eclipse Edition / `cuda-gdb`

For one GPU thread, change the value it is printing with Nsight Eclispe Edition or `cuda-gdb`

* Compile with `make`
* Run your application with `make run`
* `cuda-gdb`: Launch an interactive session on JURON with 

    ```bash
    bsub -Is -R "rusage[ngpus_shared=1]" -tty /bin/bash
    ```

    Then call `cuda-gdb app`.

    To simplify this process, use `make debug-cuda-gdb`.
* Nsight Eclipse Edition: Launch an interactive, X-forwarding session on JURON with

    ```bash
    bsub -Is -R "rusage[ngpus_shared=1]" -XF -tty /bin/bash
    ```

    Then call Nsight Eclipse Edition with `nsight`.

    To simplify this process, use `make debug-nsight`.
