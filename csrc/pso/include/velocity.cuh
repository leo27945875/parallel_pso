#pragma once

#include "utils.cuh"

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
    ssize_t       num, 
    ssize_t       dim,
    cuda_rng_t   *rng_states
);