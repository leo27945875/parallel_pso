#pragma once

#include <stdio.h>
#include <cstdlib>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

#define MAX_GRID_DIM_1D 1024UL
#define BLOCK_DIM_1D 256UL
#define BLOCK_DIM_2D 16UL

double rand_number_(double range = 1.);

__host__ __device__ size_t cdiv(size_t total, size_t size);
__host__ __device__ double pow2(double x);

__global__ void sum_rows_kernel(double *xs, double *out, size_t num, size_t dim);