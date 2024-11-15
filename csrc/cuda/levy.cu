#include <cmath>
#include "levy.cuh"
#include "utils.cuh"


__host__ __device__ double levy_w_func(double x){
    return 1. + 0.25 * (x - 1.);
}
__host__ __device__ double levy_head_func(double x){
    return pow2(sin(M_PI * levy_w_func(x)));
}
__host__ __device__ double levy_tail_func(double x){
    double w = levy_w_func(x);
    return pow2(w - 1.) * (1. + pow2(sin(2. * M_PI * w)));
}
__host__ __device__ double levy_middle_func(double x){
    double w = levy_w_func(x);
    return pow2(w - 1.) * (1. + 10. * pow2(sin(M_PI * w + 1.)));
}

__global__ void levy_function_kernel(double *xs, double *out, size_t num, size_t dim){
    __shared__ double sdata[BLOCK_DIM_1D];
    size_t nid = blockIdx.x;
    size_t bid = blockIdx.y;
    size_t tid = threadIdx.x;
    size_t idx = bid * blockDim.x + tid;

    if (nid >= num)
        return;

    sdata[tid] = 0.;
    for (size_t i = idx; i < dim; i += gridDim.y * blockDim.x){
        double x = xs[nid * dim + i];
        if (i == 0)
            sdata[tid] += levy_head_func(x) + levy_middle_func(x);
        else if (i == dim - 1)
            sdata[tid] += levy_tail_func(x);
        else
            sdata[tid] += levy_middle_func(x);
    }
    
    for (size_t k = blockDim.x / 2; k > 0; k >>= 1){
        __syncthreads();
        if (tid < k)
            sdata[tid] += sdata[tid + k];
    }
    if (tid == 0)
        out[nid * gridDim.y + bid] = sdata[0];
}

void levy_function_cuda(double *xs_cuda_ptr, double *out_cuda_ptr, size_t num, size_t dim){
    size_t num_block_per_x = min(cdiv(dim, BLOCK_DIM_1D), MAX_GRID_DIM_1D);
    dim3 grid_dims(num, num_block_per_x);
    dim3 block_dims(BLOCK_DIM_1D);
    levy_function_kernel<<<grid_dims, block_dims>>>(xs_cuda_ptr, out_cuda_ptr, num, dim); 
    cudaCheckErrors("Running 'levy_function_kernel' failed.");
    sum_rows_kernel<<<num, block_dims>>>(out_cuda_ptr, out_cuda_ptr, num, num_block_per_x); 
    cudaCheckErrors("Running 'sum_rows_kernel' of Levy function failed.");
}