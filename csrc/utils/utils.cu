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

__global__ void find_global_min_kernel(
    double const *numbers, double *global_min_num, ssize_t *global_min_idx, ssize_t size, cuda_mutex_t *mutex
){
    __shared__ double n_smem[BLOCK_DIM_1D];
    __shared__ double i_smem[BLOCK_DIM_1D];

    ssize_t tid = threadIdx.x;
    ssize_t idx = blockDim.x * blockIdx.x + tid;

    n_smem[tid] = cuda::std::numeric_limits<double>::max();
    for (ssize_t i = idx; i < size; i += gridDim.x * blockDim.x){
        double num = numbers[i];
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
__global__ void get_rand_numbers_kernel(ssize_t size, cuda_rng_t *rng_states, double *res){
    ssize_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size){
        cuda_rng_t thread_rng_state = rng_states[idx];
        res[idx] = curand_uniform_double(&thread_rng_state);
        rng_states[idx] = thread_rng_state;
    }
}
__global__ void sum_rows_kernel(double const *xs, double *out, ssize_t num, ssize_t dim){
    __shared__ double smem[BLOCK_DIM_1D];
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

__host__ double rand_number(double range){
    return (static_cast<double>(rand()) / RAND_MAX) * (2. * range) - range;
}
__host__ void curand_setup(ssize_t size, unsigned long long seed, cuda_rng_t **rng_states){
    cudaMalloc(rng_states, size * sizeof(cuda_rng_t));
    init_rng_states_kernel<<<cdiv(size, BLOCK_DIM_1D), BLOCK_DIM_1D>>>(size, *rng_states, seed);
    cudaCheckErrors("Failed to run 'init_rng_states_kernel'.");
}
__host__ void curand_destroy(cuda_rng_t *rng_states){
    cudaFree(rng_states);
    cudaCheckErrors("Failed to free CUDA memory 'rng_states'.");
}
__host__ void get_curand_numbers(ssize_t size, cuda_rng_t *rng_states, double *res){
    get_rand_numbers_kernel<<<cdiv(size, BLOCK_DIM_1D), BLOCK_DIM_1D>>>(size, rng_states, res);
    cudaCheckErrors("Failed to run 'get_rand_numbers_kernel'.");
}

__host__ void print_matrix(double const *mat, ssize_t nrow, ssize_t ncol){
    for (ssize_t i = 0; i < nrow; i++){
        for (ssize_t j = 0; j < ncol; j++)
            printf("% 8.4f, ", mat[i * ncol + j]);
        printf("\n");
    }
}
__host__ ssize_t get_num_block_1d(ssize_t dim){
    return min(MAX_NUM_BLOCK_1D, cdiv(dim, BLOCK_DIM_1D));
}
__host__ ssize_t cdiv(ssize_t total, ssize_t size){
    return (total + size - 1) / size;
}
__host__ __device__ double pow2(double x){
    return x * x;
}