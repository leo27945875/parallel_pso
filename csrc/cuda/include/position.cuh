#pragma once

void update_positions_cuda(
    double       *xs,
    double const *vs,
    double        x_min,
    double        x_max,
    size_t        num,
    size_t        dim
);