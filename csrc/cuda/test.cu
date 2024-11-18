#include <iostream>
#include <ctime>
#include <cstdlib>

#include "levy.cuh"
#include "utils.cuh"


void test_levy_function(
    size_t num         = 1000,
    size_t dim         = 500,
    double tol         = 1e-5,
    bool   is_show_out = true
){

    double *xs, *out_cpu, *out_cuda;
    double *d_xs, *d_out;

    srand(time(NULL));

    xs       = new double[num * dim];
    out_cpu  = new double[num];
    out_cuda = new double[num];

    for (size_t i = 0; i < num * dim; i++) 
        xs[i] = rand_number();
    
    levy_function_cpu(xs, out_cpu, num, dim);

    cudaMalloc(&d_xs, num * dim * sizeof(double));
    cudaMalloc(&d_out, num * sizeof(double));
    cudaMemcpy(d_xs, xs, num * dim * sizeof(double), cudaMemcpyHostToDevice);
    levy_function_cuda(d_xs, d_out, num, dim);
    cudaMemcpy(out_cuda, d_out, num * sizeof(double), cudaMemcpyDeviceToHost);

    if (is_show_out){
        std::cout << "\nLevy results: (CPU)" << std::endl;
        for (size_t i = 0; i < num; i++)
            std::cout << out_cpu[i] << ", ";
        std::cout << std::endl;

        std::cout << "\nLevy results: (CUDA)" << std::endl;
        for (size_t i = 0; i < num; i++)
            std::cout << out_cuda[i] << ", ";
        std::cout << std::endl;
    }

    bool is_close = true;
    for (size_t i = 0; i < num; i++){
        is_close = is_close && abs(out_cpu[i] - out_cuda[i]) < tol;
    }
    std::cout << "\nis_close = " << is_close << std::endl;

    cudaFree(d_xs);
    cudaFree(d_out);
    delete[] xs;
    delete[] out_cpu;
    delete[] out_cuda;
}


void test_curand(
    size_t             num    = 10,
    int                n_loop = 5,
    unsigned long long seed   = 0
){
    double *h_res, *d_res;
    curandState *d_states;
    
    h_res = new double[num];
    cudaMalloc(&d_res, num * sizeof(double));

    curand_setup(num, seed, &d_states);
    for (int loop = 0; loop < n_loop; loop++){
        get_curand_numbers(num, d_states, d_res);
        cudaMemcpy(h_res, d_res, num * sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << "cuRAND test:\n";
        for (size_t i = 0; i < num; i++)
            std::cout << h_res[i] << ", ";
        std::cout << std::endl;
    }

    cudaFree(d_res);
    cudaFree(d_states);
    delete[] h_res;
}


int main(){

    // test_levy_function();
    test_curand();
    
    return 0;
}