#include "position.cuh"
#include "utils.cuh"


__global__ void update_positions_kernel(
    double       *xs,
    double const *vs,
    double        x_min,
    double        x_max,
    ssize_t       num,
    ssize_t       dim
){
    ssize_t nid = blockIdx.x;
    ssize_t idx = blockIdx.y * blockDim.x + threadIdx.x;
    for (ssize_t i = idx; i < dim; i += gridDim.y * blockDim.x)
        xs[nid * dim + i] = min(
            x_max, 
            max(
                x_min, 
                xs[nid * dim + i] + vs[nid * dim + i]
            )
        );
}

void update_positions_cuda(
    double       *xs_cuda_ptr,
    double const *vs_cuda_ptr,
    double        x_min,
    double        x_max,
    ssize_t       num,
    ssize_t       dim
){
    dim3 grid_dims(num, get_num_block_1d(dim));
    dim3 block_dims(BLOCK_DIM_1D);
    update_positions_kernel<<<grid_dims, block_dims>>>(
        xs_cuda_ptr, vs_cuda_ptr, x_min, x_max, num, dim
    );
    cudaCheckErrors("Running 'update_positions_kernel' failed.");
}