#pragma once

#include "utils.cuh"

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
);
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
);
#endif