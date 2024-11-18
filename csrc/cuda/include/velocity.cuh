#pragma once

#include "utils.cuh"

void update_velocities_cuda(
    double       *vs_cuda_ptr, 
    double const *xs_cuda_ptr, 
    double const *local_best_xs_cuda_ptr, 
    double const *global_best_x_cuda_ptr,
    double       *sum_pow2_res,
    double        w,
    double        c0,
    double        c1,
    double        v_max,
    size_t        num, 
    size_t        dim,
    curandState   *rng_states
);