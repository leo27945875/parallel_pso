#pragma once

#include <stdio.h>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>

#define MAX_GRID_DIM_1D 1024UL
#define BLOCK_DIM_1D 256UL
#define BLOCK_DIM_2D 16UL

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "CUDA fatal error: %s (%s at %s:%d)\n", msg, cudaGetErrorString(__err), __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

#define curandCheckErrors(x, msg) \
    do { \
        if((x) != CURAND_STATUS_SUCCESS) { \
            fprintf(stderr, "cuRAND fatal error: %s (at %s:%d)\n", msg, __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while(0)


__global__ void sum_rows_kernel(double const *xs, double *out, size_t num, size_t dim);
__global__ void sum_pow2_rows_kernel(double const *xs, double *out, size_t num, size_t dim);

__host__ double rand_number(double range = 1.);
__host__ void curand_setup(size_t size, unsigned long long seed, curandState **rng_states);
__host__ void get_curand_numbers(size_t size, curandState *rng_states, double *res);

__host__ __device__ inline size_t get_num_block_per_x(size_t dim);
__host__ __device__ inline size_t cdiv(size_t total, size_t size);
__host__ __device__ inline double pow2(double x);