#pragma once

#include <stdio.h>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>

#define MAX_NUM_THREAD_PER_BLOCK 1024UL
#define MAX_GRID_DIM_1D 1024UL
#define BLOCK_DIM_1D 256UL
#define BLOCK_DIM_2D 16UL

#define IS_CHECK_CUDA_ERROR 1
#define IS_GLOBAL_BEST_USE_ATOMIC 0

// Error checking macro
#if (IS_CHECK_CUDA_ERROR)
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
#else
    #define cudaCheckErrors(msg)
    #define curandCheckErrors(x, msg)
#endif

// Mutex for kernels:
#if (IS_GLOBAL_BEST_USE_ATOMIC)
    typedef int cuda_mutex_t;
    __host__ void cuda_create_mutex(cuda_mutex_t *mutex);
    __host__ void cuda_destroy_mutex(cuda_mutex_t *mutex);
    __device__ void lock_kernel_mutex(cuda_mutex_t *mutex);
    __device__ void unlock_kernel_mutex(cuda_mutex_t *mutex);
#endif

// Other helper functions:
__host__ double rand_number(double range = 1.);
__host__ void curand_setup(size_t size, unsigned long long seed, curandState **rng_states);
__host__ void get_curand_numbers(size_t size, curandState *rng_states, double *res);

__host__ __device__ size_t get_num_block_1d(size_t dim);
__host__ __device__ size_t cdiv(size_t total, size_t size);
__host__ __device__ double pow2(double x);

__global__ void sum_rows_kernel(double const *xs, double *out, size_t num, size_t dim);