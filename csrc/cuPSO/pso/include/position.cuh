#pragma once

#include "utils.cuh"

#if not IS_CUDA_ALIGN_MALLOC
void update_positions_cuda(
    scalar_t       *xs_cuda_ptr,
    scalar_t const *vs_cuda_ptr,
    scalar_t        x_min,
    scalar_t        x_max,
    ssize_t         num,
    ssize_t         dim
);
#else
void update_positions_cuda(
    scalar_t       *xs_cuda_ptr,
    scalar_t const *vs_cuda_ptr,
    scalar_t        x_min,
    scalar_t        x_max,
    ssize_t         num,
    ssize_t         dim,
    ssize_t         xs_pitch,
    ssize_t         vs_pitch
);
#endif