#!/bin/bash

[ -z "$CUDA_HOME" ] && echo "Please load CUDA before invoking the script!" && exit 1;

make

SUBMIT='bsub -U cuda18_0 -Is -R "rusage[ngpus_shared=1]"'

export OMP_NUM_THREADS=48
$SUBMIT ./dgemm_bench.sh
$SUBMIT ./nbody_bench.sh
$SUBMIT ./ddot_bench.sh
$SUBMIT ./mandelbrot_bench.sh

make display
