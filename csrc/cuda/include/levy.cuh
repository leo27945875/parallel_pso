#pragma once

void levy_function_cuda(const double *xs_cuda_ptr, double *out_cuda_ptr, size_t num, size_t dim);
void levy_function_cpu(const double *xs, double *out, size_t num, size_t dim);