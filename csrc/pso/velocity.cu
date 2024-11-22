#include "velocity.cuh"
#include "utils.cuh"


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
    cuda_rng_t   *rng_states
){
    size_t nid = blockIdx.x;
    size_t idx = blockIdx.y * blockDim.x + threadIdx.x;

    for (size_t i = idx; i < dim; i += gridDim.y * blockDim.x){

        // Load data into local memory:
        double v = vs[nid * dim + i];
        double x = xs[nid * dim + i];
        double lbest_x = local_best_xs[nid * dim + i];
        double gbest_x = global_best_x[i];

        // Update velocities:
#if IS_VELOVITY_USE_RANDOM
        cuda_rng_t thread_rng_state = rng_states[nid * dim + i];
        vs[nid * dim + i] = (
            w * v + 
            c0 * curand_uniform_double(&thread_rng_state) * (lbest_x - x) + 
            c1 * curand_uniform_double(&thread_rng_state) * (gbest_x - x)
        );
        rng_states[nid * dim + idx] = thread_rng_state;
#else
        vs[nid * dim + i] = (
            w * v + 
            c0 * (lbest_x - x) + 
            c1 * (gbest_x - x)
        );
#endif
    }
}

__global__ void update_velocities_with_sum_pow2_kernel(
    double       *vs, 
    double const *xs, 
    double const *local_best_xs, 
    double const *global_best_x,
    double       *v_sum_pow2_res,
    double        w,
    double        c0,
    double        c1,
    size_t        num, 
    size_t        dim,
    cuda_rng_t   *rng_states
){
    __shared__ double p_smem[BLOCK_DIM_1D];

    size_t nid = blockIdx.x;
    size_t bid = blockIdx.y;
    size_t tid = threadIdx.x;
    size_t idx = bid * blockDim.x + tid;

    p_smem[tid] = 0.;
    for (size_t i = idx; i < dim; i += gridDim.y * blockDim.x){
        // Load data into local memory:
        double v = vs[nid * dim + i];
        double x = xs[nid * dim + i];
        double lbest_x = local_best_xs[nid * dim + i];
        double gbest_x = global_best_x[i];

        // Calculate new velocities:
#if IS_VELOVITY_USE_RANDOM
        cuda_rng_t thread_rng_state = rng_states[nid * dim + idx];
        v = (
            w * v + 
            c0 * curand_uniform_double(&thread_rng_state) * (lbest_x - x) + 
            c1 * curand_uniform_double(&thread_rng_state) * (gbest_x - x)
        );
        rng_states[nid * dim + idx] = thread_rng_state;
#else
        v = (
            w * v + 
            c0 * (lbest_x - x) + 
            c1 * (gbest_x - x)
        );
#endif
        // Store v^2 into shared memory:
        p_smem[tid] += v * v;

        // Store v into global memory:
        vs[nid * dim + i] = v;
    }
    // Sum the squares:
    for (size_t k = blockDim.x >> 1; k > 0; k >>= 1){
        __syncthreads();
        if (tid < k)
            p_smem[tid] += p_smem[tid + k];
    }
    if (tid == 0)
        atomicAdd(v_sum_pow2_res + nid, p_smem[0]);
}

__global__ void norm_clip_velocities_kernel(
    double *vs, 
    double *v_sum_pow2_res,
    double  v_max,
    size_t  num, 
    size_t  dim
){
    size_t nid = blockIdx.x;
    size_t idx = blockIdx.y * blockDim.x + threadIdx.x;
    double norm = sqrt(v_sum_pow2_res[nid]); // 'Broadcast' mechanism (see https://forums.developer.nvidia.com/t/accessing-same-global-memory-address-within-warps/66574)
    if (norm <= v_max)
        return;
    for (size_t i = idx; i < dim; i += gridDim.y * blockDim.x){
        vs[nid * dim + i] /= norm;
    }
}

void update_velocities_cuda(
    double       *vs_cuda_ptr, 
    double const *xs_cuda_ptr, 
    double const *local_best_xs_cuda_ptr, 
    double const *global_best_x_cuda_ptr,
    double       *v_sum_pow2_cuda_ptr,
    double        w,
    double        c0,
    double        c1,
    double        v_max,
    size_t        num, 
    size_t        dim,
    cuda_rng_t   *rng_states
){
    dim3 grid_dims(num, get_num_block_1d(dim));
    dim3 block_dims(BLOCK_DIM_1D);

    if (v_max <= 0.){
        update_velocities_kernel<<<grid_dims, block_dims>>>(
            vs_cuda_ptr, xs_cuda_ptr, local_best_xs_cuda_ptr, global_best_x_cuda_ptr, w, c0, c1, num, dim, rng_states
        );
        cudaCheckErrors("Failed to run 'update_velocities_kernel'.");
        
    }else{
        bool is_v_sum_pow2_no_buffer = (v_sum_pow2_cuda_ptr == nullptr);
        if (is_v_sum_pow2_no_buffer){
            cudaMalloc(&v_sum_pow2_cuda_ptr, num * sizeof(double));
            cudaCheckErrors("Failed to allocate memory buffer to 'v_sum_pow2_cuda_ptr'.");
        }
        cudaMemset(v_sum_pow2_cuda_ptr, 0, num * sizeof(double));
        cudaCheckErrors("Failed to set buffer 'v_sum_pow2_cuda_ptr' to zeros.");

        update_velocities_with_sum_pow2_kernel<<<grid_dims, block_dims>>>(
            vs_cuda_ptr, xs_cuda_ptr, local_best_xs_cuda_ptr, global_best_x_cuda_ptr, v_sum_pow2_cuda_ptr, w, c0, c1, num, dim, rng_states
        );
        cudaCheckErrors("Failed to run 'update_velocities_with_sum_pow2_kernel'.");
        
        norm_clip_velocities_kernel<<<grid_dims, block_dims>>>(
            vs_cuda_ptr, v_sum_pow2_cuda_ptr, v_max, num, dim
        );
        cudaCheckErrors("Failed to run 'norm_clip_velocities_kernel'.");

        if (is_v_sum_pow2_no_buffer){
            cudaFree(v_sum_pow2_cuda_ptr);
            cudaCheckErrors("Failed to free memory buffer to 'v_sum_pow2_cuda_ptr'.");
        }
    }
}