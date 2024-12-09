#include "velocity.cuh"
#include "utils.cuh"


#if not IS_CUDA_ALIGN_MALLOC
__global__ void update_velocities_kernel(
    scalar_t       *vs, 
    scalar_t const *xs, 
    scalar_t const *local_best_xs, 
    scalar_t const *global_best_x,
    scalar_t        w,
    scalar_t        c0,
    scalar_t        c1,
    ssize_t         num, 
    ssize_t         dim,
    cuda_rng_t     *rng_states
){
    ssize_t nid = blockIdx.x * blockDim.x + threadIdx.x;
    ssize_t idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (nid >= num)
        return;
        
    for (ssize_t i = idx; i < dim; i += gridDim.y * blockDim.y){
        // Load data into local memory:
        scalar_t v = vs[nid * dim + i];
        scalar_t x = xs[nid * dim + i];
        scalar_t lbest_x = local_best_xs[nid * dim + i];
        scalar_t gbest_x = global_best_x[i];

        // Update velocities:
        cuda_rng_t thread_rng_state = rng_states[nid * dim + i];
        vs[nid * dim + i] = (
            w * v + 
            c0 * _get_curand_uniform<scalar_t>(&thread_rng_state) * (lbest_x - x) + 
            c1 * _get_curand_uniform<scalar_t>(&thread_rng_state) * (gbest_x - x)
        );
        rng_states[nid * dim + i] = thread_rng_state;
    }
}
__global__ void update_velocities_with_sum_pow2_kernel(
    scalar_t       *vs, 
    scalar_t const *xs, 
    scalar_t const *local_best_xs, 
    scalar_t const *global_best_x,
    scalar_t       *v_sum_pow2_res,
    scalar_t        w,
    scalar_t        c0,
    scalar_t        c1,
    ssize_t         num, 
    ssize_t         dim,
    cuda_rng_t     *rng_states
){
    __shared__ scalar_t p_smem[BLOCK_DIM_X][BLOCK_DIM_Y];

    ssize_t nid = blockIdx.x * blockDim.x + threadIdx.x;
    ssize_t idx = blockIdx.y * blockDim.y + threadIdx.y;
    ssize_t tidx = threadIdx.x;
    ssize_t tidy = threadIdx.y;

    p_smem[tidx][tidy] = 0.;

    if (nid >= num)
        return;

    for (ssize_t i = idx; i < dim; i += gridDim.y * blockDim.y){
        // Load data into local memory:
        scalar_t v = vs[nid * dim + i];
        scalar_t x = xs[nid * dim + i];
        scalar_t lbest_x = local_best_xs[nid * dim + i];
        scalar_t gbest_x = global_best_x[i];

        // Calculate new velocities:
        cuda_rng_t thread_rng_state = rng_states[nid * dim + i];
        v = (
            w * v + 
            c0 * _get_curand_uniform<scalar_t>(&thread_rng_state) * (lbest_x - x) + 
            c1 * _get_curand_uniform<scalar_t>(&thread_rng_state) * (gbest_x - x)
        );
        rng_states[nid * dim + i] = thread_rng_state;

        // Store v^2 into shared memory:
        p_smem[tidx][tidy] += v * v;

        // Store v into global memory:
        vs[nid * dim + i] = v;
    }
    // Sum the squares:
    for (ssize_t k = blockDim.y >> 1; k > 0; k >>= 1){
        __syncthreads();
        if (tidy < k)
            p_smem[tidx][tidy] += p_smem[tidx][tidy + k];
    }
    if (tidy == 0)
        atomicAdd(v_sum_pow2_res + nid, p_smem[tidx][0]);
}
__global__ void norm_clip_velocities_kernel(
    scalar_t *vs, 
    scalar_t *v_sum_pow2_res,
    scalar_t  v_max,
    ssize_t   num, 
    ssize_t   dim
){
    ssize_t nid = blockIdx.x * blockDim.x + threadIdx.x;
    ssize_t idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (nid >= num)
        return;

    scalar_t norm = sqrt(v_sum_pow2_res[nid]); // 'Broadcast' mechanism (see https://forums.developer.nvidia.com/t/accessing-same-global-memory-address-within-warps/66574)
    if (norm <= v_max)
        return;

    scalar_t scale = v_max / norm;
    for (ssize_t i = idx; i < dim; i += gridDim.y * blockDim.y){
        vs[nid * dim + i] *= scale;
    }
}
#else
__global__ void update_velocities_aligned_kernel(
    scalar_t       *vs, 
    scalar_t const *xs, 
    scalar_t const *local_best_xs, 
    scalar_t const *global_best_x,
    scalar_t        w,
    scalar_t        c0,
    scalar_t        c1,
    ssize_t         num, 
    ssize_t         dim,
    ssize_t         vs_pitch,
    ssize_t         xs_pitch,
    ssize_t         local_best_xs_pitch,
    ssize_t         rng_states_pitch,
    cuda_rng_t     *rng_states
){
    ssize_t nid = blockIdx.x * blockDim.x + threadIdx.x;
    ssize_t idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (nid >= num)
        return;
        
    for (ssize_t i = idx; i < dim; i += gridDim.y * blockDim.y){
        // Load data into local memory:
        scalar_t v = *ptr2d_at(vs, nid, i, vs_pitch);
        scalar_t x = *ptr2d_at(xs, nid, i, xs_pitch);
        scalar_t lbest_x = *ptr2d_at(local_best_xs, nid, i, local_best_xs_pitch);
        scalar_t gbest_x = global_best_x[i];

        // Update velocities:
        cuda_rng_t thread_rng_state = *ptr2d_at(rng_states, nid, i, rng_states_pitch);
        *ptr2d_at(vs, nid, i, vs_pitch) = (
            w * v + 
            c0 * _get_curand_uniform<scalar_t>(&thread_rng_state) * (lbest_x - x) + 
            c1 * _get_curand_uniform<scalar_t>(&thread_rng_state) * (gbest_x - x)
        );
        *ptr2d_at(rng_states, nid, i, rng_states_pitch) = thread_rng_state;
    }
}
__global__ void update_velocities_with_sum_pow2_aligned_kernel(
    scalar_t       *vs, 
    scalar_t const *xs, 
    scalar_t const *local_best_xs, 
    scalar_t const *global_best_x,
    scalar_t       *v_sum_pow2_res,
    scalar_t        w,
    scalar_t        c0,
    scalar_t        c1,
    ssize_t         num, 
    ssize_t         dim,
    ssize_t         vs_pitch,
    ssize_t         xs_pitch,
    ssize_t         local_best_xs_pitch,
    ssize_t         rng_states_pitch,
    cuda_rng_t     *rng_states
){
    __shared__ scalar_t p_smem[BLOCK_DIM_X][BLOCK_DIM_Y];

    ssize_t nid = blockIdx.x * blockDim.x + threadIdx.x;
    ssize_t idx = blockIdx.y * blockDim.y + threadIdx.y;
    ssize_t tidx = threadIdx.x;
    ssize_t tidy = threadIdx.y;

    p_smem[tidx][tidy] = 0.;

    if (nid >= num)
        return;

    for (ssize_t i = idx; i < dim; i += gridDim.y * blockDim.y){
        // Load data into local memory:
        scalar_t v = *ptr2d_at(vs, nid, i, vs_pitch);
        scalar_t x = *ptr2d_at(xs, nid, i, xs_pitch);
        scalar_t lbest_x = *ptr2d_at(local_best_xs, nid, i, local_best_xs_pitch);
        scalar_t gbest_x = global_best_x[i];

        // Calculate new velocities:
        cuda_rng_t thread_rng_state = *ptr2d_at(rng_states, nid, i, rng_states_pitch);
        v = (
            w * v + 
            c0 * _get_curand_uniform<scalar_t>(&thread_rng_state) * (lbest_x - x) + 
            c1 * _get_curand_uniform<scalar_t>(&thread_rng_state) * (gbest_x - x)
        );
        *ptr2d_at(rng_states, nid, i, rng_states_pitch) = thread_rng_state;

        // Store v^2 into shared memory:
        p_smem[tidx][tidy] += v * v;

        // Store v into global memory:
        *ptr2d_at(vs, nid, i, vs_pitch) = v;
    }
    // Sum the squares:
    for (ssize_t k = blockDim.y >> 1; k > 0; k >>= 1){
        __syncthreads();
        if (tidy < k)
            p_smem[tidx][tidy] += p_smem[tidx][tidy + k];
    }
    if (tidy == 0)
        atomicAdd(v_sum_pow2_res + nid, p_smem[tidx][0]);
}
__global__ void norm_clip_velocities_aligned_kernel(
    scalar_t *vs, 
    scalar_t *v_sum_pow2_res,
    scalar_t  v_max,
    ssize_t   num, 
    ssize_t   dim,
    ssize_t   vs_pitch
){
    ssize_t nid = blockIdx.x * blockDim.x + threadIdx.x;
    ssize_t idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (nid >= num)
        return;

    scalar_t norm = sqrt(v_sum_pow2_res[nid]); // 'Broadcast' mechanism (see https://forums.developer.nvidia.com/t/accessing-same-global-memory-address-within-warps/66574)
    if (norm <= v_max)
        return;

    scalar_t scale = v_max / norm;
    for (ssize_t i = idx; i < dim; i += gridDim.y * blockDim.y){
        *ptr2d_at(vs, nid, i, vs_pitch) *= scale;
    }
}
#endif

#if not IS_CUDA_ALIGN_MALLOC
void update_velocities_cuda(
    scalar_t       *vs_cuda_ptr, 
    scalar_t const *xs_cuda_ptr, 
    scalar_t const *local_best_xs_cuda_ptr, 
    scalar_t const *global_best_x_cuda_ptr,
    scalar_t       *v_sum_pow2_cuda_ptr,
    scalar_t        w,
    scalar_t        c0,
    scalar_t        c1,
    scalar_t        v_max,
    ssize_t         num, 
    ssize_t         dim,
    cuda_rng_t     *rng_states_cuda_ptr
){
    dim3 grid_dims(get_num_block_x(num), get_num_block_y(dim));
    dim3 block_dims(BLOCK_DIM_X, BLOCK_DIM_Y);

    if (v_max <= 0.){
        update_velocities_kernel<<<grid_dims, block_dims>>>(
            vs_cuda_ptr, xs_cuda_ptr, local_best_xs_cuda_ptr, global_best_x_cuda_ptr, w, c0, c1, num, dim, rng_states_cuda_ptr
        );
        cudaCheckErrors("Failed to run 'update_velocities_kernel'.");
        
    }else{
        bool is_v_sum_pow2_no_buffer = (v_sum_pow2_cuda_ptr == nullptr);
        if (is_v_sum_pow2_no_buffer){
            cudaMalloc(&v_sum_pow2_cuda_ptr, num * sizeof(scalar_t));
            cudaCheckErrors("Failed to allocate memory buffer to 'v_sum_pow2_cuda_ptr'.");
        }
        cudaMemset(v_sum_pow2_cuda_ptr, 0, num * sizeof(scalar_t));
        cudaCheckErrors("Failed to zero out buffer 'v_sum_pow2_cuda_ptr'.");

        update_velocities_with_sum_pow2_kernel<<<grid_dims, block_dims>>>(
            vs_cuda_ptr, xs_cuda_ptr, local_best_xs_cuda_ptr, global_best_x_cuda_ptr, v_sum_pow2_cuda_ptr, w, c0, c1, num, dim, rng_states_cuda_ptr
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
#else
void update_velocities_cuda(
    scalar_t       *vs_cuda_ptr, 
    scalar_t const *xs_cuda_ptr, 
    scalar_t const *local_best_xs_cuda_ptr, 
    scalar_t const *global_best_x_cuda_ptr,
    scalar_t       *v_sum_pow2_cuda_ptr,
    scalar_t        w,
    scalar_t        c0,
    scalar_t        c1,
    scalar_t        v_max,
    ssize_t         num, 
    ssize_t         dim,
    ssize_t         vs_pitch,
    ssize_t         xs_pitch,
    ssize_t         local_best_xs_pitch,
    ssize_t         rng_states_pitch,
    cuda_rng_t     *rng_states_cuda_ptr
){
    dim3 grid_dims(get_num_block_x(num), get_num_block_y(dim));
    dim3 block_dims(BLOCK_DIM_X, BLOCK_DIM_Y);

    if (v_max <= 0.){
        update_velocities_aligned_kernel<<<grid_dims, block_dims>>>(
            vs_cuda_ptr, xs_cuda_ptr, local_best_xs_cuda_ptr, global_best_x_cuda_ptr, w, c0, c1, 
            num, dim, vs_pitch, xs_pitch, local_best_xs_pitch, rng_states_pitch, rng_states_cuda_ptr
        );
        cudaCheckErrors("Failed to run 'update_velocities_aligned_kernel'.");
        
    }else{
        bool is_v_sum_pow2_no_buffer = (v_sum_pow2_cuda_ptr == nullptr);
        if (is_v_sum_pow2_no_buffer){
            cudaMalloc(&v_sum_pow2_cuda_ptr, num * sizeof(scalar_t));
            cudaCheckErrors("Failed to allocate memory buffer to 'v_sum_pow2_cuda_ptr'.");
        }
        cudaMemset(v_sum_pow2_cuda_ptr, 0, num * sizeof(scalar_t));
        cudaCheckErrors("Failed to zero out set buffer 'v_sum_pow2_cuda_ptr'.");

        update_velocities_with_sum_pow2_aligned_kernel<<<grid_dims, block_dims>>>(
            vs_cuda_ptr, xs_cuda_ptr, local_best_xs_cuda_ptr, global_best_x_cuda_ptr, v_sum_pow2_cuda_ptr, w, c0, c1, num, dim, 
            vs_pitch, xs_pitch, local_best_xs_pitch, rng_states_pitch, rng_states_cuda_ptr
        );
        cudaCheckErrors("Failed to run 'update_velocities_with_sum_pow2_aligned_kernel'.");
        
        norm_clip_velocities_aligned_kernel<<<grid_dims, block_dims>>>(vs_cuda_ptr, v_sum_pow2_cuda_ptr, v_max, num, dim, vs_pitch);
        cudaCheckErrors("Failed to run 'norm_clip_velocities_aligned_kernel'.");

        if (is_v_sum_pow2_no_buffer){
            cudaFree(v_sum_pow2_cuda_ptr);
            cudaCheckErrors("Failed to free memory buffer to 'v_sum_pow2_cuda_ptr'.");
        }
    }
}
#endif