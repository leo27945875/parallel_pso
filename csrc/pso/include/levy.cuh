#pragma once

void levy_function_cuda(double const *xs_cuda_ptr, double *out_cuda_ptr, size_t num, size_t dim);
void levy_function_cpu(double const *xs, double *out, size_t num, size_t dim);