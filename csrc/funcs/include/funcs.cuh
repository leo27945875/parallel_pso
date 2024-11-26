#pragma once


double levy(double const *x, ssize_t dim);
void levy_function_cuda(double const *xs_cuda_ptr, double *out_cuda_ptr, ssize_t num, ssize_t dim);
void levy_function_cpu(double const *xs, double *out, ssize_t num, ssize_t dim);
