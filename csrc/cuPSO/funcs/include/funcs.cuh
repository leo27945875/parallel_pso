#pragma once

#include "utils.cuh"

scalar_t levy(scalar_t const *x, ssize_t dim);
void levy_function_cpu(scalar_t const *xs, scalar_t *out, ssize_t num, ssize_t dim);
void levy_function_cuda(scalar_t const *xs_cuda_ptr, scalar_t *out_cuda_ptr, ssize_t num, ssize_t dim);
void levy_function_cuda(scalar_t const *xs_cuda_ptr, scalar_t *out_cuda_ptr, ssize_t num, ssize_t dim, ssize_t xs_pitch);
