#include <cmath>
#include <vector>
#include "funcs.cuh"
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

__global__ void levy_function_kernel(double const *xs, double *out, ssize_t num, ssize_t dim){
    __shared__ double smem[BLOCK_DIM_1D];
    ssize_t nid = blockIdx.x;
    ssize_t bid = blockIdx.y;
    ssize_t tid = threadIdx.x;
    ssize_t idx = bid * blockDim.x + tid;

    if (nid >= num)
        return;

    smem[tid] = 0.;
    for (ssize_t i = idx; i < dim; i += gridDim.y * blockDim.x){
        double x = xs[nid * dim + i];
        if (i == 0)
            smem[tid] += levy_head_func(x) + levy_middle_func(x);
        else if (i == dim - 1)
            smem[tid] += levy_tail_func(x);
        else
            smem[tid] += levy_middle_func(x);
    }
    
    for (ssize_t k = blockDim.x >> 1; k > 0; k >>= 1){
        __syncthreads();
        if (tid < k)
            smem[tid] += smem[tid + k];
    }
    if (tid == 0)
        atomicAdd(out + nid, smem[0]);
}


double levy(double const *x, ssize_t dim){
    double out = 0.;
    for (ssize_t idx = 0; idx < dim; idx++){
        double _x = x[idx];
        if (idx == 0)
            out += levy_head_func(_x) + levy_middle_func(_x);
        else if (idx == dim - 1)
            out += levy_tail_func(_x);
        else
            out += levy_middle_func(_x);
    }
    return out;
}


void levy_function_cuda(double const *xs_cuda_ptr, double *out_cuda_ptr, ssize_t num, ssize_t dim){
    ssize_t num_block_per_x = get_num_block_1d(dim);
    dim3 grid_dims(num, num_block_per_x);
    dim3 block_dims(BLOCK_DIM_1D);
    cudaMemset(out_cuda_ptr, 0, num * sizeof(double));
    cudaCheckErrors("Failed to zero out buffer 'out_cuda_ptr'.");
    levy_function_kernel<<<grid_dims, block_dims>>>(xs_cuda_ptr, out_cuda_ptr, num, dim); 
    cudaCheckErrors("Failed to run 'levy_function_kernel'.");
}

void levy_function_cpu(double const *xs, double *out, ssize_t num, ssize_t dim){
    for (ssize_t nid = 0; nid < num; nid++)
        out[nid] = levy(xs + nid * dim, dim);
}