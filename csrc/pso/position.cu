#include "position.cuh"
#include "utils.cuh"


__global__ void update_positions_kernel(
    scalar_t       *xs,
    scalar_t const *vs,
    scalar_t        x_min,
    scalar_t        x_max,
    ssize_t         num,
    ssize_t         dim
){
    ssize_t nid = blockIdx.x * blockDim.x + threadIdx.x;
    ssize_t idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (nid >= num)
        return;
    for (ssize_t i = idx; i < dim; i += gridDim.y * blockDim.y)
        xs[nid * dim + i] = min(
            x_max, 
            max(
                x_min, 
                xs[nid * dim + i] + vs[nid * dim + i]
            )
        );
}

void update_positions_cuda(
    scalar_t       *xs_cuda_ptr,
    scalar_t const *vs_cuda_ptr,
    scalar_t        x_min,
    scalar_t        x_max,
    ssize_t         num,
    ssize_t         dim
){
    dim3 grid_dims(get_num_block_x(num), get_num_block_y(dim));
    dim3 block_dims(BLOCK_DIM_X, BLOCK_DIM_Y);
    update_positions_kernel<<<grid_dims, block_dims>>>(
        xs_cuda_ptr, vs_cuda_ptr, x_min, x_max, num, dim
    );
    cudaCheckErrors("Failed to run 'update_positions_kernel'.");
}