#include "utils.cuh"


__global__ void init_rng_states_kernel(curandState *states, unsigned long long seed){
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    curand_init(seed, idx, 0, states + idx);
}

__global__ void sum_rows_kernel(double const *xs, double *out, size_t num, size_t dim){
    __shared__ double smem[BLOCK_DIM_1D];
    int nid = blockIdx.x;
    int bid = blockIdx.y;
    int tid = threadIdx.x;
    int idx = bid * blockDim.x + tid;

    if (nid >= num)
        return;
    
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

__global__ void sum_pow2_rows_kernel(double const *xs, double *out, size_t num, size_t dim){
    __shared__ double smem[BLOCK_DIM_1D];
    int nid = blockIdx.x;
    int bid = blockIdx.y;
    int tid = threadIdx.x;
    int idx = bid * blockDim.x + tid;

    if (nid >= num)
        return;
    
    smem[tid] = 0.;
    for (size_t i = idx; i < dim; i += gridDim.y * blockDim.x)
        smem[tid] += pow2(xs[nid * dim + i]);
    
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
__host__ void curand_setup(size_t size, unsigned long long seed, curandState **states){
    cudaMalloc(states, size * sizeof(curandState));
    init_rng_states_kernel<<<cdiv(size, BLOCK_DIM_1D), BLOCK_DIM_1D>>>(*states, seed);
}


__host__ size_t get_num_block_per_x(size_t dim){
    return min(cdiv(dim, BLOCK_DIM_1D), MAX_GRID_DIM_1D);
}


__host__ __device__ size_t cdiv(size_t total, size_t size){
    return (total + size - 1) / size;
}
__host__ __device__ double pow2(double x){
    return x * x;
}