#include <cuda/std/limits>
#include "utils.cuh"


__host__ void cuda_create_mutex(cuda_mutex_t **mutex){
    cudaMalloc(mutex, sizeof(cuda_mutex_t));
    cudaMemset(*mutex, 0, sizeof(cuda_mutex_t));
}
__host__ void cuda_destroy_mutex(cuda_mutex_t *mutex){
    cudaFree(mutex);
}
__device__ void lock_kernel_mutex(cuda_mutex_t *mutex){
    while (atomicCAS(mutex, 0, 1) != 0);
}
__device__ void unlock_kernel_mutex(cuda_mutex_t *mutex){
    atomicExch(mutex, 0);
}


__host__ void print_matrix(scalar_t const *mat, ssize_t nrow, ssize_t ncol){
    for (ssize_t i = 0; i < nrow; i++){
        for (ssize_t j = 0; j < ncol; j++)
            printf("% 8.4f, ", mat[i * ncol + j]);
        printf("\n");
    }
}
__host__ void print_cuda_matrix(scalar_t const *mat, ssize_t nrow, ssize_t ncol){
    scalar_t *tmp = new scalar_t[nrow * ncol];
    cudaMemcpy(tmp, mat, nrow * ncol * sizeof(scalar_t), cudaMemcpyDeviceToHost);
    cudaCheckErrors("Failed to copy matrix to host.");
    print_matrix(tmp, nrow, ncol);
    delete[] tmp;
}
__host__ ssize_t cdiv(ssize_t total, ssize_t size){
    return (total + size - 1) / size;
}
__host__ ssize_t get_num_block_x(ssize_t dim){
    return cdiv(dim, BLOCK_DIM_X);
}
__host__ ssize_t get_num_block_y(ssize_t dim){
    return min(MAX_NUM_BLOCK_1D, cdiv(dim, BLOCK_DIM_Y));
}

__global__ void sum_rows_kernel(scalar_t const *xs, scalar_t *out, ssize_t num, ssize_t dim){
    __shared__ scalar_t smem[BLOCK_DIM];
    int nid = blockIdx.x;
    int bid = blockIdx.y;
    int tid = threadIdx.x;
    int idx = bid * blockDim.x + tid;

    smem[tid] = 0.;
    for (ssize_t i = idx; i < dim; i += gridDim.y * blockDim.x)
        smem[tid] += xs[nid * dim + i];
    
    for (ssize_t k = blockDim.x >> 1; k > 0; k >>= 1){
        __syncthreads();
        if (tid < k)
            smem[tid] += smem[tid + k];
    }
    if (tid == 0)
        atomicAdd(out + nid, smem[0]);
}

__global__ void find_global_min_kernel(
    scalar_t const *numbers, scalar_t *global_min_num, ssize_t *global_min_idx, ssize_t size, cuda_mutex_t *mutex
){
    __shared__ scalar_t n_smem[BLOCK_DIM];
    __shared__ scalar_t i_smem[BLOCK_DIM];

    ssize_t tid = threadIdx.x;
    ssize_t idx = blockDim.x * blockIdx.x + tid;

    n_smem[tid] = cuda::std::numeric_limits<scalar_t>::max();
    for (ssize_t i = idx; i < size; i += gridDim.x * blockDim.x){
        scalar_t num = numbers[i];
        if (num < n_smem[tid]){
            n_smem[tid] = num;
            i_smem[tid] = i;
        }
    }
    for (ssize_t k = blockDim.x >> 1; k > 0; k >>= 1){
        __syncthreads();
        if (tid < k)
        if (n_smem[tid + k] < n_smem[tid]){
            n_smem[tid] = n_smem[tid + k];
            i_smem[tid] = i_smem[tid + k];
        }
    }
    if (tid == 0){
        lock_kernel_mutex(mutex);
        if (n_smem[0] < *global_min_num){
            *global_min_num = n_smem[0];
            *global_min_idx = i_smem[0];
        }
        unlock_kernel_mutex(mutex);
    }
}

__global__ void init_rng_states_kernel(ssize_t size, cuda_rng_t *states, unsigned long long seed){
    ssize_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size)
        curand_init(seed, idx, 0, states + idx);
}
__global__ void init_rng_states_kernel(ssize_t nrow, ssize_t ncol, ssize_t pitch, cuda_rng_t *states, unsigned long long seed){
    ssize_t nid = blockDim.x * blockIdx.x + threadIdx.x;
    ssize_t idx = blockDim.y * blockIdx.y + threadIdx.y;
    for (ssize_t i = idx; i < ncol; i += gridDim.y * blockDim.y)
        curand_init(seed, idx, 0, ptr2d_at(states, nid, i, pitch));
}

__global__ void get_rand_numbers_kernel(ssize_t size, cuda_rng_t *rng_states, scalar_t *res){
    ssize_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size){
        cuda_rng_t thread_rng_state = rng_states[idx];
        res[idx] = _get_curand_uniform<scalar_t>(&thread_rng_state);
        rng_states[idx] = thread_rng_state;
    }
}
__global__ void get_rand_numbers_kernel(ssize_t nrow, ssize_t ncol, cuda_rng_t *rng_states, scalar_t *res, ssize_t rng_pitch, ssize_t res_pitch){
    ssize_t nid = blockDim.x * blockIdx.x + threadIdx.x;
    ssize_t idx = blockDim.y * blockIdx.y + threadIdx.y;
    for (ssize_t i = idx; i < ncol; i += gridDim.y * blockDim.y){
        cuda_rng_t thread_rng_state = *ptr2d_at(rng_states, nid, i, rng_pitch);
        *ptr2d_at(res, nid, i, res_pitch) = _get_curand_uniform<scalar_t>(&thread_rng_state);
        *ptr2d_at(rng_states, nid, i, rng_pitch) = thread_rng_state;
    }
}

__host__ scalar_t rand_number(scalar_t range){
    return (static_cast<scalar_t>(rand()) / RAND_MAX) * (2. * range) - range;
}

__host__ void curand_setup(ssize_t size, unsigned long long seed, cuda_rng_t **rng_states){
    cudaMalloc(rng_states, size * sizeof(cuda_rng_t));
    init_rng_states_kernel<<<cdiv(size, BLOCK_DIM), BLOCK_DIM>>>(size, *rng_states, seed);
    cudaCheckErrors("Failed to run 'init_rng_states_kernel'.");
}
__host__ void curand_setup(ssize_t nrow, ssize_t ncol, unsigned long long seed, cuda_rng_t **rng_states, ssize_t *pitch){
#if IS_CUDA_ALIGN_MALLOC
    cudaMallocPitch(rng_states, reinterpret_cast<size_t *>(pitch), ncol * sizeof(cuda_rng_t), nrow);
#else
    cudaMalloc(rng_states, nrow * ncol * sizeof(cuda_rng_t));
    *pitch = ncol * sizeof(cuda_rng_t);
#endif
    dim3 grid_dims(get_num_block_x(nrow), get_num_block_y(ncol));
    dim3 block_dims(BLOCK_DIM_X, BLOCK_DIM_Y);
    init_rng_states_kernel<<<grid_dims, block_dims>>>(nrow, ncol, *pitch, *rng_states, seed);
    cudaCheckErrors("Failed to run 'init_rng_states_kernel'.");
}

__host__ void curand_destroy(cuda_rng_t *rng_states){
    cudaFree(rng_states);
    cudaCheckErrors("Failed to free CUDA memory 'rng_states'.");
}

__host__ void get_curand_numbers(ssize_t size, cuda_rng_t *rng_states, scalar_t *res){
    get_rand_numbers_kernel<<<cdiv(size, BLOCK_DIM), BLOCK_DIM>>>(size, rng_states, res);
    cudaCheckErrors("Failed to run 'get_rand_numbers_kernel'.");
}
__host__ void get_curand_numbers(ssize_t nrow, ssize_t ncol, cuda_rng_t *rng_states, scalar_t *res, ssize_t rng_pitch, ssize_t res_pitch){
    dim3 grid_dims(get_num_block_x(nrow), get_num_block_y(ncol));
    dim3 block_dims(BLOCK_DIM_X, BLOCK_DIM_Y);
    get_rand_numbers_kernel<<<grid_dims, block_dims>>>(nrow, ncol, rng_states, res, rng_pitch, res_pitch);
    cudaCheckErrors("Failed to run 'get_rand_numbers_kernel'.");
}