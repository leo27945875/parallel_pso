#pragma once

size_t update_best_fits_cuda(
    double const *x_fits_cuda_ptr,
    double       *local_best_fits_cuda_ptr,
    double       *global_best_fit_cuda_ptr,
    size_t        num
);

size_t update_bests_cuda(
    double const *xs_cuda_ptr,
    double const *x_fits_cuda_ptr,
    double       *local_best_xs_cuda_ptr,
    double       *local_best_fits_cuda_ptr,
    double       *global_best_x_cuda_ptr,
    double       *global_best_fit_cuda_ptr,
    size_t        num,
    size_t        dim
);