#pragma once

#include <stdio.h>

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


__host__ __device__ bool is_even(size_t i){
    return i % 2 == 0;
}
__host__ __device__ size_t cdiv(size_t total, size_t size){
    return (total + size - 1) / size;
}
__host__ __device__ double pow2(double x){
    return x * x;
}

__global__ void sum_rows_kernel(double *xs, double *out, size_t num, size_t dim){
    __shared__ double sdata[BLOCK_DIM_1D];
    int nid = blockIdx.x;
    int bid = blockIdx.y;
    int tid = threadIdx.x;
    int idx = bid * blockDim.x + tid;

    if (nid >= num)
        return;
    
    sdata[tid] = 0.;
    for (size_t i = idx; i < dim; i += gridDim.y * blockDim.x)
        sdata[tid] += xs[nid * dim + i];
    
    for (size_t k = blockDim.x / 2; k > 0; k >>= 1){
        __syncthreads();
        if (tid < k)
            sdata[tid] += sdata[tid + k];
    }
    if (tid == 0)
        out[nid * gridDim.y + bid] = sdata[0];
}
