#include "utils.cuh"

#if (IS_GLOBAL_BEST_USE_ATOMIC)
    __host__ void cuda_create_mutex(cuda_mutex_t *mutex){
        cudaMemset(mutex, 0, sizeof(cuda_mutex_t));
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
#endif


__global__ void init_rng_states_kernel(size_t size, curandState *states, unsigned long long seed){
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size)
        curand_init(seed, idx, 0, states + idx);
}
__global__ void get_rand_numbers_kernel(size_t size, curandState *rng_states, double *res){
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size){
        curandState thread_rng_state = rng_states[idx];
        res[idx] = curand_uniform_double(&thread_rng_state);
        rng_states[idx] = thread_rng_state;
    }
}
__global__ void sum_rows_kernel(double const *xs, double *out, size_t num, size_t dim){
    __shared__ double smem[BLOCK_DIM_1D];
    int nid = blockIdx.x;
    int bid = blockIdx.y;
    int tid = threadIdx.x;
    int idx = bid * blockDim.x + tid;

    smem[tid] = 0.;
    for (size_t i = idx; i < dim; i += gridDim.y * blockDim.x)
        smem[tid] += xs[nid * dim + i];
    
    for (size_t k = blockDim.x / 2; k > 0; k >>= 1){
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
__host__ void curand_setup(size_t size, unsigned long long seed, curandState **rng_states){
    cudaMalloc(rng_states, size * sizeof(curandState));
    init_rng_states_kernel<<<cdiv(size, BLOCK_DIM_1D), BLOCK_DIM_1D>>>(size, *rng_states, seed);
    cudaCheckErrors("Running 'init_rng_states_kernel' failed.");
}
__host__ void get_curand_numbers(size_t size, curandState *rng_states, double *res){
    get_rand_numbers_kernel<<<cdiv(size, BLOCK_DIM_1D), BLOCK_DIM_1D>>>(size, rng_states, res);
    cudaCheckErrors("Running 'get_rand_numbers_kernel' failed.");
}


__host__ __device__ size_t get_num_block_1d(size_t dim){
    return min(cdiv(dim, BLOCK_DIM_1D), MAX_GRID_DIM_1D);
}
__host__ __device__ size_t cdiv(size_t total, size_t size){
    return (total + size - 1) / size;
}
__host__ __device__ double pow2(double x){
    return x * x;
}