#!/bin/sh
# Build libkilosort_kernels.so next to this script. Needs the CUDA toolkit (nvcc on PATH).
set -e
cd "$(dirname "$0")"
nvcc -O3 -shared -Xcompiler -fPIC -o libkilosort_kernels.so kilosort_kernels.cu \
    -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120 \
    -gencode arch=compute_120,code=compute_120
