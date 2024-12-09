#pragma once

#include "utils.cuh"

ssize_t update_best_fits_cuda(
    scalar_t const *x_fits_cuda_ptr,
    scalar_t       *local_best_fits_cuda_ptr,
    scalar_t       *global_best_fit_cuda_ptr,
    ssize_t         num
);

#if not IS_CUDA_ALIGN_MALLOC
ssize_t update_bests_cuda( 
    scalar_t const *xs_cuda_ptr,
    scalar_t const *x_fits_cuda_ptr,
    scalar_t       *local_best_xs_cuda_ptr,
    scalar_t       *local_best_fits_cuda_ptr,
    scalar_t       *global_best_x_cuda_ptr,
    scalar_t       *global_best_fit_cuda_ptr,
    ssize_t         num,
    ssize_t         dim
);
#else
ssize_t update_bests_cuda(
    scalar_t const *xs_cuda_ptr,
    scalar_t const *x_fits_cuda_ptr,
    scalar_t       *local_best_xs_cuda_ptr,
    scalar_t       *local_best_fits_cuda_ptr,
    scalar_t       *global_best_x_cuda_ptr,
    scalar_t       *global_best_fit_cuda_ptr,
    ssize_t         num,
    ssize_t         dim,
    ssize_t         xs_pitch,
    ssize_t         local_best_xs_pitch
);
#endif