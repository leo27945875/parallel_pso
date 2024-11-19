#include <iostream>
#include <ctime>
#include <cfloat>
#include <cstdlib>

#include "levy.cuh"
#include "evolve.cuh"
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


void test_update_best(
    size_t num = 200
){

    srand(time(NULL));

    double *h_x_fits, *h_local_best_fits, *h_global_best_fit, *x_fits;
    double *d_x_fits, *d_local_best_fits, *d_global_best_fit;
    size_t global_best_idx = num;

    double real_global_min_fit = DBL_MAX;
    size_t real_global_min_idx = num;

    h_x_fits = new double[num];
    h_local_best_fits = new double[num];
    h_global_best_fit = new double;
    x_fits = new double[num];
    
    for (size_t i = 0; i < num; i++){
        h_x_fits[i] = rand_number(static_cast<double>(num));
        h_local_best_fits[i] = static_cast<double>(num / 2);
        if (h_x_fits[i] < real_global_min_fit){
            real_global_min_fit = h_x_fits[i];
            real_global_min_idx = i;
        }
        x_fits[i] = h_x_fits[i];
    }
    *h_global_best_fit = DBL_MAX;


    std::cout << "Before:";
    // std::cout << "\nx_fits: ";
    // for (size_t i = 0; i < num; i++) std::cout << h_x_fits[i] << ", ";
    // std::cout << "\nl_fits: ";
    // for (size_t i = 0; i < num; i++) std::cout << h_local_best_fits[i] << ", ";
    std::cout << "\ng_fits: ";
    std::cout << *h_global_best_fit << " | idx = " << global_best_idx << " (real = " << real_global_min_fit << " | idx = " << real_global_min_idx << ")" << std::endl;

    cudaMalloc(&d_x_fits         , num * sizeof(double));
    cudaMalloc(&d_local_best_fits, num * sizeof(double));
    cudaMalloc(&d_global_best_fit,       sizeof(double));

    cudaMemcpy(d_x_fits         , h_x_fits         , num * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_local_best_fits, h_local_best_fits, num * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_global_best_fit, h_global_best_fit,       sizeof(double), cudaMemcpyHostToDevice);

    global_best_idx = update_best_fits_cuda(
        d_x_fits, d_local_best_fits, d_global_best_fit, num
    );

    cudaMemcpy(h_x_fits         , d_x_fits         , num * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_local_best_fits, d_local_best_fits, num * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_global_best_fit, d_global_best_fit,       sizeof(double), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    std::cout << "\nAfter:";
    // std::cout << "\nx_fits: ";
    // for (size_t i = 0; i < num; i++) std::cout << h_x_fits[i] << ", ";
    // std::cout << "\nl_fits: ";
    // for (size_t i = 0; i < num; i++) std::cout << h_local_best_fits[i] << ", ";
    std::cout << "\ng_fits: ";
    std::cout << *h_global_best_fit << " | idx = " << global_best_idx << " (real = " << real_global_min_fit << " | idx = " << real_global_min_idx << ")" << std::endl;

    std::cout << "\nCheck race condition:" << std::endl;
    std::cout << "Real: " << x_fits[real_global_min_idx] << " | " << h_x_fits[real_global_min_idx] << std::endl;
    std::cout << "CUDA: " << x_fits[global_best_idx] << " | " << h_x_fits[global_best_idx] << std::endl;

    cudaFree(d_x_fits);
    cudaFree(d_local_best_fits);
    cudaFree(d_global_best_fit);
    delete[] h_x_fits;
    delete[] h_local_best_fits;
    delete[] h_global_best_fit;
    delete[] x_fits;
}


int main(){

    // test_levy_function();
    // test_curand();
    test_update_best();
    cudaDeviceSynchronize();
    return 0;
}