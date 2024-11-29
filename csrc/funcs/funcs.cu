#include <cmath>
#include <vector>
#include "funcs.cuh"
#include "utils.cuh"


__host__ __device__ scalar_t levy_w_func(scalar_t x){
    return 1. + 0.25 * (x - 1.);
}
__host__ __device__ scalar_t levy_head_func(scalar_t x){
    return pow2(sin(M_PI * levy_w_func(x)));
}
__host__ __device__ scalar_t levy_tail_func(scalar_t x){
    scalar_t w = levy_w_func(x);
    return pow2(w - 1.) * (1. + pow2(sin(2. * M_PI * w)));
}
__host__ __device__ scalar_t levy_middle_func(scalar_t x){
    scalar_t w = levy_w_func(x);
    return pow2(w - 1.) * (1. + 10. * pow2(sin(M_PI * w + 1.)));
}


__global__ void levy_function_kernel(scalar_t const *xs, scalar_t *out, ssize_t num, ssize_t dim){
    __shared__ scalar_t smem[BLOCK_DIM_X][BLOCK_DIM_Y];
    ssize_t nid = blockIdx.x * blockDim.x + threadIdx.x;
    ssize_t idx = blockIdx.y * blockDim.y + threadIdx.y;
    ssize_t tidx = threadIdx.x;
    ssize_t tidy = threadIdx.y;

    smem[tidx][tidy] = 0.;

    if (nid >= num)
        return;
        
    for (ssize_t i = idx; i < dim; i += gridDim.y * blockDim.y){
        scalar_t x = xs[nid * dim + i];
        if (i == 0)
            smem[tidx][tidy] += levy_head_func(x) + levy_middle_func(x);
        else if (i == dim - 1)
            smem[tidx][tidy] += levy_tail_func(x);
        else
            smem[tidx][tidy] += levy_middle_func(x);
    }
    for (ssize_t k = blockDim.y >> 1; k > 0; k >>= 1){
        __syncthreads();
        if (tidy < k)
            smem[tidx][tidy] += smem[tidx][tidy + k];
    }
    if (tidy == 0)
        atomicAdd(out + nid, smem[tidx][0]);
}
__global__ void levy_function_aligned_kernel(scalar_t const *xs, scalar_t *out, ssize_t num, ssize_t dim, ssize_t xs_pitch){
    __shared__ scalar_t smem[BLOCK_DIM_X][BLOCK_DIM_Y];
    ssize_t nid = blockIdx.x * blockDim.x + threadIdx.x;
    ssize_t idx = blockIdx.y * blockDim.y + threadIdx.y;
    ssize_t tidx = threadIdx.x;
    ssize_t tidy = threadIdx.y;

    smem[tidx][tidy] = 0.;

    if (nid >= num)
        return;
    
    for (ssize_t i = idx; i < dim; i += gridDim.y * blockDim.y){
        scalar_t x = *ptr2d_at(xs, nid, i, xs_pitch);
        if (i == 0)
            smem[tidx][tidy] += levy_head_func(x) + levy_middle_func(x);
        else if (i == dim - 1)
            smem[tidx][tidy] += levy_tail_func(x);
        else
            smem[tidx][tidy] += levy_middle_func(x);
    }
    for (ssize_t k = blockDim.y >> 1; k > 0; k >>= 1){
        __syncthreads();
        if (tidy < k)
            smem[tidx][tidy] += smem[tidx][tidy + k];
    }
    if (tidy == 0)
        atomicAdd(out + nid, smem[tidx][0]);
}


scalar_t levy(scalar_t const *x, ssize_t dim){
    scalar_t out = 0.;
    for (ssize_t idx = 0; idx < dim; idx++){
        scalar_t _x = x[idx];
        if (idx == 0)
            out += levy_head_func(_x) + levy_middle_func(_x);
        else if (idx == dim - 1)
            out += levy_tail_func(_x);
        else
            out += levy_middle_func(_x);
    }
    return out;
}

void levy_function_cpu(scalar_t const *xs, scalar_t *out, ssize_t num, ssize_t dim){
    for (ssize_t nid = 0; nid < num; nid++)
        out[nid] = levy(xs + nid * dim, dim);
}

void levy_function_cuda(scalar_t const *xs_cuda_ptr, scalar_t *out_cuda_ptr, ssize_t num, ssize_t dim){
    dim3 grid_dims(get_num_block_x(num), get_num_block_y(dim));
    dim3 block_dims(BLOCK_DIM_X, BLOCK_DIM_Y);
    cudaMemset(out_cuda_ptr, 0, num * sizeof(scalar_t));
    cudaCheckErrors("Failed to zero out buffer 'out_cuda_ptr'.");
    levy_function_kernel<<<grid_dims, block_dims>>>(xs_cuda_ptr, out_cuda_ptr, num, dim); 
    cudaCheckErrors("Failed to run 'levy_function_kernel'.");
}
void levy_function_cuda(scalar_t const *xs_cuda_ptr, scalar_t *out_cuda_ptr, ssize_t num, ssize_t dim, ssize_t xs_pitch){
    dim3 grid_dims(get_num_block_x(num), get_num_block_y(dim));
    dim3 block_dims(BLOCK_DIM_X, BLOCK_DIM_Y);
    cudaMemset(out_cuda_ptr, 0, num * sizeof(scalar_t));
    cudaCheckErrors("Failed to zero out buffer 'out_cuda_ptr'.");
    levy_function_aligned_kernel<<<grid_dims, block_dims>>>(xs_cuda_ptr, out_cuda_ptr, num, dim, xs_pitch); 
    cudaCheckErrors("Failed to run 'levy_function_kernel'.");
}