#include "utils.cuh"


double rand_number_(double range){
    return (static_cast<double>(rand()) / RAND_MAX) * (2. * range) - range;
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