#include "position.cuh"
#include "utils.cuh"


__global__ void update_positions_kernel(
    double       *xs,
    double const *vs,
    double        x_min,
    double        x_max,
    size_t        num,
    size_t        dim
){
    size_t nid = blockIdx.x;
    size_t idx = blockIdx.y * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < dim; i += gridDim.y * blockDim.x)
        xs[nid * dim + i] = min(
            x_max, 
            max(
                x_min, 
                xs[nid * dim + i] + vs[nid * dim + i]
            )
        );
}

void update_positions_cuda(
    double       *xs,
    double const *vs,
    double        x_min,
    double        x_max,
    size_t        num,
    size_t        dim
){
    size_t num_block_per_x = get_num_block_per_x(dim);
    dim3 grid_dims(num, num_block_per_x);
    dim3 block_dims(BLOCK_DIM_1D);
    update_positions_kernel<<<grid_dims, block_dims>>>(
        xs, vs, x_min, x_max, num, dim
    );
    cudaCheckErrors("Running 'update_positions_kernel' failed.");
}