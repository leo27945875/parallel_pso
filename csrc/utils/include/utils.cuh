#pragma once

#include <cstdio>
#include <cstdlib>
#include <curand_kernel.h>

#define MAX_NUM_BLOCK_1D 128L
#define BLOCK_DIM_X      1L
#define BLOCK_DIM_Y      256L
#define BLOCK_DIM        (BLOCK_DIM_X * BLOCK_DIM_Y)

#define IS_TESTING                0
#define IS_CHECK_CUDA_ERROR       1
#define IS_CUDA_ALIGN_MALLOC      0
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
#else
    #define cudaCheckErrors(msg)
#endif

// Float precision:
typedef double scalar_t;

// Mutex for kernels:
typedef int cuda_mutex_t;
__host__   void cuda_create_mutex  (cuda_mutex_t **mutex);
__host__   void cuda_destroy_mutex (cuda_mutex_t  *mutex);
__device__ void lock_kernel_mutex  (cuda_mutex_t  *mutex);
__device__ void unlock_kernel_mutex(cuda_mutex_t  *mutex);

// Random number functions:
typedef curandState cuda_rng_t;
__host__ scalar_t rand_number       (scalar_t range = 1.);
__host__ void     curand_setup      (ssize_t size, unsigned long long seed, cuda_rng_t **rng_states);
__host__ void     curand_setup      (ssize_t nrow, ssize_t ncol, unsigned long long seed, cuda_rng_t **rng_states, ssize_t *pitch);
__host__ void     curand_destroy    (cuda_rng_t *rng_states);
__host__ void     get_curand_numbers(ssize_t size, cuda_rng_t *rng_states, scalar_t *res);
__host__ void     get_curand_numbers(ssize_t nrow, ssize_t ncol, cuda_rng_t *rng_states, scalar_t *res, ssize_t rng_pitch, ssize_t res_pitch);

template<typename T> static __forceinline__ __device__ T      _get_curand_uniform(cuda_rng_t *rng_state);
template<>                  __forceinline__ __device__ float  _get_curand_uniform(cuda_rng_t *rng_state){ return curand_uniform(rng_state); }
template<>                  __forceinline__ __device__ double _get_curand_uniform(cuda_rng_t *rng_state){ return curand_uniform_double(rng_state); }

// Other helper functions:
__host__ void print_matrix     (scalar_t const *mat, ssize_t nrow, ssize_t ncol);
__host__ void print_cuda_matrix(scalar_t const *mat, ssize_t nrow, ssize_t ncol);

__host__ ssize_t cdiv           (ssize_t total, ssize_t size);
__host__ ssize_t get_num_block_x(ssize_t dim);
__host__ ssize_t get_num_block_y(ssize_t dim);

static __forceinline__ __host__ __device__ scalar_t           pow2    (scalar_t x){ return x * x; }
static __forceinline__ __host__ __device__ scalar_t         * ptr2d_at(scalar_t         *mat, ssize_t row, ssize_t col, ssize_t pitch){ return (scalar_t   *)((char *)mat + row * pitch + col * sizeof(scalar_t)  ); }
static __forceinline__ __host__ __device__ scalar_t   const * ptr2d_at(scalar_t   const *mat, ssize_t row, ssize_t col, ssize_t pitch){ return (scalar_t   *)((char *)mat + row * pitch + col * sizeof(scalar_t)  ); }
static __forceinline__ __host__ __device__ cuda_rng_t       * ptr2d_at(cuda_rng_t       *mat, ssize_t row, ssize_t col, ssize_t pitch){ return (cuda_rng_t *)((char *)mat + row * pitch + col * sizeof(cuda_rng_t)); }
static __forceinline__ __host__ __device__ cuda_rng_t const * ptr2d_at(cuda_rng_t const *mat, ssize_t row, ssize_t col, ssize_t pitch){ return (cuda_rng_t *)((char *)mat + row * pitch + col * sizeof(cuda_rng_t)); }

__global__ void sum_rows_kernel       (scalar_t const *xs, scalar_t *out, ssize_t num, ssize_t dim);
__global__ void find_global_min_kernel(scalar_t const *numbers, scalar_t *global_min_num, ssize_t *global_min_idx, ssize_t size, cuda_mutex_t *mutex);