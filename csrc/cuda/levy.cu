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

__global__ void levy_function_kernel(double const *xs, double *out, size_t num, size_t dim){
    __shared__ double smem[BLOCK_DIM_1D];
    size_t nid = blockIdx.x;
    size_t bid = blockIdx.y;
    size_t tid = threadIdx.x;
    size_t idx = bid * blockDim.x + tid;

    if (nid >= num)
        return;

    smem[tid] = 0.;
    for (size_t i = idx; i < dim; i += gridDim.y * blockDim.x){
        double x = xs[nid * dim + i];
        if (i == 0)
            smem[tid] += levy_head_func(x) + levy_middle_func(x);
        else if (i == dim - 1)
            smem[tid] += levy_tail_func(x);
        else
            smem[tid] += levy_middle_func(x);
    }
    
    for (size_t k = blockDim.x / 2; k > 0; k >>= 1){
        __syncthreads();
        if (tid < k)
            smem[tid] += smem[tid + k];
    }
    if (tid == 0)
        atomicAdd(out + nid, smem[0]);
}

void levy_function_cuda(double const *xs_cuda_ptr, double *out_cuda_ptr, size_t num, size_t dim){
    size_t num_block_per_x = get_num_block_1d(dim);
    dim3 grid_dims(num, num_block_per_x);
    dim3 block_dims(BLOCK_DIM_1D);
    levy_function_kernel<<<grid_dims, block_dims>>>(xs_cuda_ptr, out_cuda_ptr, num, dim); 
    cudaCheckErrors("Running 'levy_function_kernel' failed.");
}

void levy_function_cpu(double const *xs, double *out, size_t num, size_t dim){
    for (size_t nid = 0; nid < num; nid++){
        out[nid] = 0.;
        for (size_t idx = 0; idx < dim; idx++){
            double x = xs[nid * dim + idx];
            if (idx == 0)
                out[nid] += levy_head_func(x) + levy_middle_func(x);
            else if (idx == dim - 1)
                out[nid] += levy_tail_func(x);
            else
                out[nid] += levy_middle_func(x);
        }
    }
}