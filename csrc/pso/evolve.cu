#include <cuda/std/limits>

#include "evolve.cuh"
#include "utils.cuh"


#if IS_GLOBAL_BEST_USE_ATOMIC
__global__ void update_best_fits_atomic_kernel(
    double const *x_fits,
    double       *local_best_fits,
    double       *glboal_best_fit,
    size_t       *global_best_idx,
    size_t        num,
    cuda_mutex_t *mutex
){
    __shared__ double glboal_best_fits_smem[BLOCK_DIM_1D];
    __shared__ size_t global_best_idxs_smem[BLOCK_DIM_1D];

    size_t tid = threadIdx.x;
    size_t idx = blockDim.x * blockIdx.x + tid;
    glboal_best_fits_smem[tid] = cuda::std::numeric_limits<double>::max();
    
    for (size_t i = idx; i < num; i += gridDim.x * blockDim.x){
        double x_fit = x_fits[i];
        local_best_fits[i] = min(x_fit, local_best_fits[i]);
        if (x_fit < glboal_best_fits_smem[tid]){
            glboal_best_fits_smem[tid] = x_fit;
            global_best_idxs_smem[tid] = i;
        }
    }
    for (size_t k = blockDim.x >> 1; k > 0; k >>= 1){
        __syncthreads();
        if (tid < k){
            if (glboal_best_fits_smem[tid + k] < glboal_best_fits_smem[tid]){
                glboal_best_fits_smem[tid] = glboal_best_fits_smem[tid + k];
                global_best_idxs_smem[tid] = global_best_idxs_smem[tid + k];
            }
        }
    }
    if (tid == 0){
        lock_kernel_mutex(mutex);
        if (glboal_best_fits_smem[0] < *glboal_best_fit){
            *glboal_best_fit = glboal_best_fits_smem[0];
            *global_best_idx = global_best_idxs_smem[0];
        }
        unlock_kernel_mutex(mutex);
    }
}

#else

__global__ void update_best_fits_reduce_kernel(
    double const *x_fits,
    double       *local_best_fits,
    double       *glboal_best_fits,
    size_t       *global_best_idxs,
    size_t        num
){
    __shared__ double glboal_best_fits_smem[BLOCK_DIM_1D];
    __shared__ size_t global_best_idxs_smem[BLOCK_DIM_1D];

    size_t bid = blockIdx.x;
    size_t tid = threadIdx.x;
    size_t idx = blockDim.x * bid + tid;
    glboal_best_fits_smem[tid] = cuda::std::numeric_limits<double>::max();
    
    for (size_t i = idx; i < num; i += gridDim.x * blockDim.x){
        double x_fit = x_fits[i];
        local_best_fits[i] = min(x_fit, local_best_fits[i]);
        if (x_fit < glboal_best_fits_smem[tid]){
            glboal_best_fits_smem[tid] = x_fit;
            global_best_idxs_smem[tid] = i;
        }
    }
    for (size_t k = blockDim.x >> 1; k > 0; k >>= 1){
        __syncthreads();
        if (tid < k){
            if (glboal_best_fits_smem[tid + k] < glboal_best_fits_smem[tid]){
                glboal_best_fits_smem[tid] = glboal_best_fits_smem[tid + k];
                global_best_idxs_smem[tid] = global_best_idxs_smem[tid + k];
            }
        }
    }
    if (tid == 0){
        glboal_best_fits[bid] = glboal_best_fits_smem[0];
        global_best_idxs[bid] = global_best_idxs_smem[0];
    }
}

__global__ void argmin_global_fits_reduce_kernel(
    double const *glboal_best_fits,
    size_t const *global_best_idxs,
    double       *glboal_best_fit,
    size_t       *global_best_idx,
    size_t        num
){
    __shared__ double glboal_best_fits_smem[BLOCK_DIM_1D];
    __shared__ size_t global_best_idxs_smem[BLOCK_DIM_1D];
    
    size_t tid = threadIdx.x;
    glboal_best_fits_smem[tid] = cuda::std::numeric_limits<double>::max();

    for (size_t i = tid; i < num; i += blockDim.x){
        double part_fit = glboal_best_fits[i];
        size_t part_idx = global_best_idxs[i];
        if (part_fit < glboal_best_fits_smem[tid]){
            glboal_best_fits_smem[tid] = part_fit;
            global_best_idxs_smem[tid] = part_idx;
        }
    }
    for (size_t k = blockDim.x >> 1; k > 0; k >>= 1){
        if (tid < k){
            if (glboal_best_fits_smem[tid + k] < glboal_best_fits_smem[tid]){
                glboal_best_fits_smem[tid] = glboal_best_fits_smem[tid + k];
                global_best_idxs_smem[tid] = global_best_idxs_smem[tid + k];
            }
        }
    }
    if (tid == 0){
        *glboal_best_fit = glboal_best_fits_smem[0];
        *global_best_idx = global_best_idxs_smem[0];
    }
}

#endif

__global__ void assign_local_best_xs(
    double const *xs,
    double const *x_fits,
    double       *local_best_xs,
    double const *local_best_fits,
    size_t        num,
    size_t        dim
){
    size_t nid = blockIdx.x;
    size_t idx = blockDim.x * blockIdx.y + threadIdx.x;

    if (x_fits[nid] > local_best_fits[nid])
        return;

    for (size_t i = idx; i < dim; i += gridDim.y * blockDim.x){
        local_best_xs[nid * dim + i] = xs[nid * dim + i];
    }
}


size_t update_best_fits_cuda(
    double const *x_fits_cuda_ptr,
    double       *local_best_fits_cuda_ptr,
    double       *global_best_fit_cuda_ptr,
    size_t        num
){
    size_t num_block_1d = get_num_block_1d(num);
    size_t global_best_idx;
    size_t *global_best_idx_cuda_ptr;

    cudaMalloc(&global_best_idx_cuda_ptr, sizeof(size_t));
    cudaCheckErrors("Failed to allocate memory buffer to 'global_best_idx_cuda_ptr'.");

#if IS_GLOBAL_BEST_USE_ATOMIC
        cuda_mutex_t *mutex;
        cuda_create_mutex(&mutex);
        cudaCheckErrors("Failed to create kernel mutex.");
        update_best_fits_atomic_kernel<<<num_block_1d, BLOCK_DIM_1D>>>(x_fits_cuda_ptr, local_best_fits_cuda_ptr, global_best_fit_cuda_ptr, global_best_idx_cuda_ptr, num, mutex);
        cudaCheckErrors("Failed to run 'update_best_fits_kernel'.");
        cuda_destroy_mutex(mutex);
        cudaCheckErrors("Failed to destroy kernel mutex.");
#else
        if (num_block_1d == 1){
            update_best_fits_reduce_kernel<<<1, BLOCK_DIM_1D>>>(x_fits_cuda_ptr, local_best_fits_cuda_ptr, global_best_fit_cuda_ptr, global_best_idx_cuda_ptr, num);
            cudaCheckErrors("Failed to run 'update_best_fits_reduce_kernel'.");
        }else{
            double *part_global_best_fits_cuda_ptr;
            size_t *part_global_best_idxs_cuda_ptr;
            cudaMalloc(&part_global_best_fits_cuda_ptr, num_block_1d * sizeof(double)); cudaCheckErrors("Failed to allocate memory buffer 'part_global_best_fits_cuda_ptr'.");
            cudaMalloc(&part_global_best_idxs_cuda_ptr, num_block_1d * sizeof(size_t)); cudaCheckErrors("Failed to allocate memory buffer 'part_global_best_idxs_cuda_ptr'.");
            update_best_fits_reduce_kernel<<<num_block_1d, BLOCK_DIM_1D>>>(x_fits_cuda_ptr, local_best_fits_cuda_ptr, part_global_best_fits_cuda_ptr, part_global_best_idxs_cuda_ptr, num); 
            cudaCheckErrors("Failed to run 'update_best_fits_reduce_kernel'.");
            argmin_global_fits_reduce_kernel<<<1, BLOCK_DIM_1D>>>(part_global_best_fits_cuda_ptr, part_global_best_idxs_cuda_ptr, global_best_fit_cuda_ptr, global_best_idx_cuda_ptr, num); 
            cudaCheckErrors("Failed to run 'argmin_global_fits_reduce_kernel'.");
            cudaFree(part_global_best_fits_cuda_ptr); cudaCheckErrors("Failed to free 'part_global_best_fits_cuda_ptr'.");
            cudaFree(part_global_best_idxs_cuda_ptr); cudaCheckErrors("Failed to free 'part_global_best_idxs_cuda_ptr'.");
        }
#endif

    cudaMemcpy(&global_best_idx, global_best_idx_cuda_ptr, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaCheckErrors("Failed to copy data from 'global_best_idx_cuda_ptr'.");
    cudaFree(global_best_idx_cuda_ptr);
    cudaCheckErrors("Failed to free 'global_best_idx_cuda_ptr'.");
    return global_best_idx;
}

size_t update_bests_cuda(
    double const *xs_cuda_ptr,
    double const *x_fits_cuda_ptr,
    double       *local_best_xs_cuda_ptr,
    double       *local_best_fits_cuda_ptr,
    double       *global_best_x_cuda_ptr,
    double       *global_best_fit_cuda_ptr,
    size_t        num,
    size_t        dim
){
    // Update the local & global best fitnesses and find the index of the global best x:
    size_t global_best_idx = update_best_fits_cuda(x_fits_cuda_ptr, local_best_fits_cuda_ptr, global_best_fit_cuda_ptr, num);

    // Assign the local best xs according to the updated local best fitnesses:
    dim3 grid_dims(num, get_num_block_1d(dim));
    dim3 block_dims(BLOCK_DIM_1D);
    assign_local_best_xs<<<grid_dims, block_dims>>>(xs_cuda_ptr, x_fits_cuda_ptr, local_best_xs_cuda_ptr, local_best_fits_cuda_ptr, num, dim);
    cudaCheckErrors("Fail to run 'assign_local_best_xs'.");

    // Assign the global best x according to the index of the global best x:
    cudaMemcpy(global_best_x_cuda_ptr, xs_cuda_ptr + global_best_idx * dim, dim * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaCheckErrors("Fail to copy global best x.");
    return global_best_idx;
}