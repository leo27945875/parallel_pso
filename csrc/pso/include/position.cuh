#pragma once

void update_positions_cuda(
    double       *xs_cuda_ptr,
    double const *vs_cuda_ptr,
    double        x_min,
    double        x_max,
    ssize_t       num,
    ssize_t       dim
);