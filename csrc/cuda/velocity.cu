#include "velocity.cuh"


__global__ void update_velocities_kernel(
    double       *vs, 
    double const *xs, 
    double const *local_best_xs, 
    double const *global_best_x,
    double        w,
    double        c0,
    double        c1,
    size_t        num, 
    size_t        dim,
    curandState   *rng_states
){
    __shared__ double v_smem[BLOCK_DIM_1D];
    __shared__ double x_smem[BLOCK_DIM_1D];

    size_t nid = blockIdx.x;
    size_t bid = blockIdx.y;
    size_t tid = threadIdx.x;
    size_t idx = bid * blockDim.x + tid;

    curandState thread_rng_state = rng_states[nid * dim + idx];

    for (size_t i = idx; i < dim; i += gridDim.y * blockDim.x){
        // Load data into shared memory:
        double v = vs[nid * dim + i];
        double x = xs[nid * dim + i];
        double lbest_x = local_best_xs[nid * dim + i];
        double gbest_x = global_best_x[i];

        // Calculate new velocities:
        vs[nid * dim + i] = (
            w * v + 
            c0 * curand_uniform_double(&thread_rng_state) * (lbest_x - x) + 
            c1 * curand_uniform_double(&thread_rng_state) * (gbest_x - x)
        );
    }
    rng_states[nid * dim + idx] = thread_rng_state;
}


void update_velocities_cuda(
    double       *vs_cuda_ptr, 
    double const *xs_cuda_ptr, 
    double const *local_best_xs_cuda_ptr, 
    double const *global_best_x_cuda_ptr,
    double        w,
    double        c0,
    double        c1,
    double        v_max,
    size_t        num, 
    size_t        dim,
    curandState   *rng_states
){
    size_t num_block_per_x = get_num_block_per_x(dim);
    dim3 grid_dims(num, num_block_per_x);
    dim3 block_dims(BLOCK_DIM_1D);
    update_velocities_kernel<<<grid_dims, block_dims>>>(
        vs_cuda_ptr, xs_cuda_ptr, local_best_xs_cuda_ptr, global_best_x_cuda_ptr, w, c0, c1, v_max, num, dim, rng_states
    );
    cudaCheckErrors("Running 'update_velocities_kernel' failed.");
}