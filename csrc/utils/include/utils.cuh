#pragma once

#include <cstdio>
#include <cstdlib>
#include <curand_kernel.h>

#define MAX_NUM_BLOCK_1D 128UL
#define BLOCK_DIM_1D     256UL

#define IS_TESTING                0
#define IS_CHECK_CUDA_ERROR       1
#define IS_VELOVITY_USE_RANDOM    1
#define IS_GLOBAL_BEST_USE_ATOMIC 1

// Error checking macro
#if IS_CHECK_CUDA_ERROR
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
typedef int cuda_mutex_t;
__host__ void cuda_create_mutex(cuda_mutex_t **mutex);
__host__ void cuda_destroy_mutex(cuda_mutex_t *mutex);
__device__ void lock_kernel_mutex(cuda_mutex_t *mutex);
__device__ void unlock_kernel_mutex(cuda_mutex_t *mutex);

// Random number functions:
typedef curandState cuda_rng_t;
__host__ double rand_number(double range = 1.);
__host__ void curand_setup(ssize_t size, unsigned long long seed, cuda_rng_t **rng_states);
__host__ void curand_destroy(cuda_rng_t *rng_states);
__host__ void get_curand_numbers(ssize_t size, cuda_rng_t *rng_states, double *res);

// Other helper functions:
__host__ void print_matrix(double const *mat, ssize_t nrow, ssize_t ncol);

__host__ ssize_t get_num_block_1d(ssize_t dim);
__host__ ssize_t cdiv(ssize_t total, ssize_t size);

__global__ void sum_rows_kernel(double const *xs, double *out, ssize_t num, ssize_t dim);
__global__ void find_global_min_kernel(double const *numbers, double *global_min_num, ssize_t *global_min_idx, ssize_t size, cuda_mutex_t *mutex);