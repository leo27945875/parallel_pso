#include <iostream>
#include "levy.cuh"

int main(){

    size_t num = 8;
    size_t dim = 2;

    double *xs, *out;
    double *d_xs, *d_out;

    xs = new double[num * dim];
    out = new double[num];
    for (size_t i = 0; i < num * dim; i++) 
        xs[i] = 1.;
    for (size_t i = 0; i < num; i++) 
        out[i] = 0.;

    xs[1 * dim + 0] = 10.; xs[1 * dim + 1] = 5.;
    
    cudaMalloc(&d_xs, num * dim * sizeof(double));
    cudaMalloc(&d_out, num * sizeof(double));
    cudaMemcpy(d_xs, xs, num * dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, xs, num * sizeof(double), cudaMemcpyHostToDevice);

    levy_function_cuda(d_xs, d_out, num, dim);

    cudaMemcpy(out, d_out, num * sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "Levy results:" << std::endl;
    for (size_t i = 0; i < num; i++)
        std::cout << out[i] << ", ";
    std::cout << std::endl;

    return 0;
}