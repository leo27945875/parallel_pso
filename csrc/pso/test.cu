#if IS_TESTING

#include <iostream>
#include <ctime>
#include <cfloat>
#include <cstdlib>

#include "levy.cuh"
#include "velocity.cuh"
#include "position.cuh"
#include "evolve.cuh"
#include "utils.cuh"


void test_curand(
    ssize_t            num    = 10,
    int                n_loop = 5,
    unsigned long long seed   = 0
){
    double *h_res, *d_res;
    cuda_rng_t *d_states;
    
    h_res = new double[num];
    cudaMalloc(&d_res, num * sizeof(double));

    curand_setup(num, seed, &d_states);
    for (int loop = 0; loop < n_loop; loop++){
        get_curand_numbers(num, d_states, d_res);
        cudaMemcpy(h_res, d_res, num * sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << "cuRAND test:\n";
        for (ssize_t i = 0; i < num; i++)
            std::cout << h_res[i] << ", ";
        std::cout << std::endl;
    }

    cudaFree(d_res);
    cudaFree(d_states);
    delete[] h_res;
}


void test_mutex(
    ssize_t num = 100033
){
    srand(time(NULL));

    double *h_arr, *d_arr;

    double h_global_min_num = DBL_MAX, real_global_min_num = DBL_MAX;
    ssize_t h_global_min_idx = num, real_global_min_idx = num;

    double *d_global_min_num;
    ssize_t *d_global_min_idx;

    h_arr = new double[num];
    for (ssize_t i = 0; i < num; i++){
        h_arr[i] = rand_number(10.);
        if (h_arr[i] < real_global_min_num){
            real_global_min_num = h_arr[i];
            real_global_min_idx = i;
        }
    }
    
    cudaMalloc(&d_arr, num * sizeof(double));
    cudaMalloc(&d_global_min_num, sizeof(double));
    cudaMalloc(&d_global_min_idx, sizeof(ssize_t));

    cudaMemcpy(d_arr, h_arr, num * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_global_min_num, &h_global_min_num, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_global_min_idx, &h_global_min_idx, sizeof(ssize_t), cudaMemcpyHostToDevice);

    cuda_mutex_t *mutex;
    cuda_create_mutex(&mutex);
    find_global_min_kernel<<<get_num_block_1d(num), BLOCK_DIM_1D>>>(d_arr, d_global_min_num, d_global_min_idx, num, mutex);
    cuda_destroy_mutex(mutex);

    cudaMemcpy(&h_global_min_num, d_global_min_num, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_global_min_idx, d_global_min_idx, sizeof(ssize_t), cudaMemcpyDeviceToHost);

    for (ssize_t i = 0; i < num; i++)
        std::cout << h_arr[i] << std::endl;
    std::cout << "\nREAL:\n";
    std::cout << "MIN_NUM = " << real_global_min_num << " | MIN_IDX = " << real_global_min_idx << std::endl;
    std::cout << "\nCUDA:\n";
    std::cout << "MIN_NUM = " << h_global_min_num << " | MIN_IDX = " << h_global_min_idx << std::endl;

    cudaFree(d_arr);
    cudaFree(d_global_min_num);
    cudaFree(d_global_min_idx);
    delete[] h_arr;
}


void test_levy_function(
    ssize_t num        = 1000,
    ssize_t dim        = 500,
    double tol         = 1e-5,
    bool   is_show_out = true
){

    double *xs, *out_cpu, *out_cuda;
    double *d_xs, *d_out;

    srand(time(NULL));

    xs       = new double[num * dim];
    out_cpu  = new double[num];
    out_cuda = new double[num];

    for (ssize_t i = 0; i < num * dim; i++) 
        xs[i] = rand_number();
    
    levy_function_cpu(xs, out_cpu, num, dim);

    cudaMalloc(&d_xs, num * dim * sizeof(double));
    cudaMalloc(&d_out, num * sizeof(double));
    cudaMemcpy(d_xs, xs, num * dim * sizeof(double), cudaMemcpyHostToDevice);
    levy_function_cuda(d_xs, d_out, num, dim);
    cudaMemcpy(out_cuda, d_out, num * sizeof(double), cudaMemcpyDeviceToHost);

    if (is_show_out){
        std::cout << "\nLevy results: (CPU)" << std::endl;
        for (ssize_t i = 0; i < num; i++)
            std::cout << out_cpu[i] << ", ";
        std::cout << std::endl;

        std::cout << "\nLevy results: (CUDA)" << std::endl;
        for (ssize_t i = 0; i < num; i++)
            std::cout << out_cuda[i] << ", ";
        std::cout << std::endl;
    }

    bool is_close = true;
    for (ssize_t i = 0; i < num; i++){
        is_close = is_close && abs(out_cpu[i] - out_cuda[i]) < tol;
    }
    std::cout << "\nis_close = " << is_close << std::endl;

    cudaFree(d_xs);
    cudaFree(d_out);
    delete[] xs;
    delete[] out_cpu;
    delete[] out_cuda;
}


void test_update_velocity(
    ssize_t             num           = 10,
    ssize_t             dim           = 500,
    double             v_max          = 1.,
    bool               is_norm_buffer = true,
    unsigned long long seed           = 0
){
    cuda_rng_t *d_states;
    double *h_vs, *h_xs, *h_local_best_xs, *h_global_best_x, *h_norm_buffer;
    double *d_vs, *d_xs, *d_local_best_xs, *d_global_best_x, *d_norm_buffer;

    curand_setup(num * dim, seed, &d_states);

    h_vs            = new double[num * dim];
    h_xs            = new double[num * dim];
    h_local_best_xs = new double[num * dim];
    h_global_best_x = new double[dim];
    h_norm_buffer   = new double[num];

    for (ssize_t i = 0; i < num; i++){
        h_norm_buffer[i] = 0.;
        for (ssize_t j = 0; j < dim; j++){
            h_vs           [i * dim + j] = 0.;
            h_xs           [i * dim + j] = static_cast<double>(i);
            h_local_best_xs[i * dim + j] = static_cast<double>(num / 2);
            h_global_best_x[          j] = static_cast<double>(num / 2);
        }
    }

    std::cout << "Before:\n";
    std::cout << "xs:\n"    ; print_matrix(h_xs, num, dim)           ; std::cout << std::endl;
    std::cout << "ls:\n"    ; print_matrix(h_local_best_xs, num, dim); std::cout << std::endl;
    std::cout << "gs:\n"    ; print_matrix(h_global_best_x, 1, dim)  ; std::cout << std::endl;
    std::cout << "vs:\n"    ; print_matrix(h_vs, num, dim)           ; std::cout << std::endl;
    std::cout << std::endl;

    cudaMalloc(&d_vs           , num * dim * sizeof(double));
    cudaMalloc(&d_xs           , num * dim * sizeof(double));
    cudaMalloc(&d_local_best_xs, num * dim * sizeof(double));
    cudaMalloc(&d_global_best_x,       dim * sizeof(double));
    cudaMalloc(&d_norm_buffer  , num       * sizeof(double));

    cudaMemcpy(d_vs           , h_vs           , num * dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xs           , h_xs           , num * dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_local_best_xs, h_local_best_xs, num * dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_global_best_x, h_global_best_x,       dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_norm_buffer  , h_norm_buffer  , num       * sizeof(double), cudaMemcpyHostToDevice);

    update_velocities_cuda(
        d_vs, d_xs, d_local_best_xs, d_global_best_x, (is_norm_buffer? d_norm_buffer: nullptr), 
        1., 1., 1., v_max, num, dim, d_states
    );

    cudaMemcpy(h_vs           , d_vs           , num * dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_norm_buffer  , d_norm_buffer  , num       * sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "After:\n";
    std::cout << "vs:\n"     ; print_matrix(h_vs, num, dim)       ; std::cout << std::endl;
    std::cout << "norm^2:\n" ; print_matrix(h_norm_buffer, num, 1); std::cout << std::endl;

    cudaFree(d_states);
    cudaFree(d_vs);
    cudaFree(d_xs);
    cudaFree(d_local_best_xs);
    cudaFree(d_global_best_x);
    cudaFree(d_norm_buffer);
    delete[] h_xs;
    delete[] h_vs;
    delete[] h_local_best_xs;
    delete[] h_global_best_x;
    delete[] h_norm_buffer;
}


void test_update_position(
    ssize_t num   = 10,
    ssize_t dim   = 500,
    double  x_min = -7.,
    double  x_max = 7.
){
    double *h_xs, *h_vs;
    double *d_xs, *d_vs;

    h_xs = new double[num * dim];
    h_vs = new double[num * dim];

    memset(h_xs, 0, num * dim * sizeof(double));
    for (ssize_t i = 0; i < num; i++){
        for (ssize_t j = 0; j < dim; j++){
            h_vs[i * dim + j] = -static_cast<double>(num) + 2 * static_cast<double>(i);
        }
    }

    std::cout << "Before:\n";
    std::cout << "xs:\n"    ; print_matrix(h_xs, num, dim);
    std::cout << "vs:\n"    ; print_matrix(h_vs, num, dim);
    std::cout << std::endl;

    cudaMalloc(&d_xs, num * dim * sizeof(double));
    cudaMalloc(&d_vs, num * dim * sizeof(double));

    cudaMemcpy(d_xs, h_xs, num * dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vs, h_vs, num * dim * sizeof(double), cudaMemcpyHostToDevice);

    update_positions_cuda(d_xs, d_vs, x_min, x_max, num, dim);

    cudaMemcpy(h_xs, d_xs, num * dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vs, d_vs, num * dim * sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "After:\n";
    std::cout << "xs:\n"   ; print_matrix(h_xs, num, dim);

    cudaFree(d_xs);
    cudaFree(d_vs);
    delete[] h_xs;
    delete[] h_vs;
}


void test_update_best(
    ssize_t num = 10000,
    ssize_t dim = 5000
){

    srand(time(NULL));

    double *h_xs, *h_local_best_xs, *h_global_best_x;
    double *h_x_fits, *h_local_best_fits, *h_global_best_fit, *x_fits;

    double *d_xs, *d_local_best_xs, *d_global_best_x;
    double *d_x_fits, *d_local_best_fits, *d_global_best_fit;

    ssize_t real_global_min_idx = num;
    double real_global_min_fit = DBL_MAX;

    ssize_t global_best_idx = num;

    h_xs            = new double[num * dim];
    h_local_best_xs = new double[num * dim];
    h_global_best_x = new double[dim];

    for (ssize_t i = 0; i < num; i++)
    for (ssize_t j = 0; j < dim; j++){
        h_xs[i * dim + j] = static_cast<double>(i + 1);
        h_local_best_xs[i * dim + j] = 0.;
        h_global_best_x[j] = 0.;
    }

    h_x_fits          = new double[num];
    h_local_best_fits = new double[num];
    h_global_best_fit = new double;
    x_fits            = new double[num];
    
    for (ssize_t i = 0; i < num; i++){
        h_x_fits[i] = rand_number(num); // static_cast<double>(i);
        h_local_best_fits[i] = static_cast<double>(num / 2);
        if (h_x_fits[i] < real_global_min_fit){
            real_global_min_fit = h_x_fits[i];
            real_global_min_idx = i;
        }
        x_fits[i] = h_x_fits[i]; // record the initial fits.
    }
    *h_global_best_fit = DBL_MAX;


    std::cout << "Before:";
    // std::cout << "\nl_xs:\n"  ; print_matrix(h_local_best_xs, num, dim);
    // std::cout << "\ng_x:\n"   ; print_matrix(h_global_best_x, 1, dim);
    // std::cout << "\nx_fits:\n"; print_matrix(h_x_fits, num, 1);
    // std::cout << "\nl_fits:\n"; print_matrix(h_local_best_fits, num, 1);
    std::cout << "\ng_fits:\n"; std::cout << *h_global_best_fit << " | idx = " << global_best_idx << " (real = " << real_global_min_fit << " | idx = " << real_global_min_idx << ")" << std::endl;

    cudaMalloc(&d_xs           , num * dim * sizeof(double));
    cudaMalloc(&d_local_best_xs, num * dim * sizeof(double));
    cudaMalloc(&d_global_best_x,       dim * sizeof(double));

    cudaMemcpy(d_xs           , h_xs           , num * dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_local_best_xs, h_local_best_xs, num * dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_global_best_x, h_global_best_x,       dim * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&d_x_fits         , num * sizeof(double));
    cudaMalloc(&d_local_best_fits, num * sizeof(double));
    cudaMalloc(&d_global_best_fit,       sizeof(double));

    cudaMemcpy(d_x_fits         , h_x_fits         , num * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_local_best_fits, h_local_best_fits, num * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_global_best_fit, h_global_best_fit,       sizeof(double), cudaMemcpyHostToDevice);

    global_best_idx = update_bests_cuda(
        d_xs, d_x_fits, d_local_best_xs, d_local_best_fits, d_global_best_x, d_global_best_fit, num, dim
    );

    cudaMemcpy(h_xs           , d_xs           , num * dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_local_best_xs, d_local_best_xs, num * dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_global_best_x, d_global_best_x,       dim * sizeof(double), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_x_fits         , d_x_fits         , num * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_local_best_fits, d_local_best_fits, num * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_global_best_fit, d_global_best_fit,       sizeof(double), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    std::cout << "\nAfter:";
    // std::cout << "\nl_xs:\n"  ; print_matrix(h_local_best_xs, num, dim);
    // std::cout << "\ng_x:\n"   ; print_matrix(h_global_best_x, 1, dim);
    // std::cout << "\nx_fits:\n"; print_matrix(h_x_fits, num, 1);
    // std::cout << "\nl_fits:\n"; print_matrix(h_local_best_fits, num, 1);
    std::cout << "\ng_fits:\n"; std::cout << *h_global_best_fit << " | idx = " << global_best_idx << " (real = " << real_global_min_fit << " | idx = " << real_global_min_idx << ")" << std::endl;

    std::cout << "\nCheck race condition:" << std::endl;
    std::cout << "Real: " << x_fits[real_global_min_idx] << " | " << h_x_fits[real_global_min_idx] << std::endl;
    std::cout << "CUDA: " << x_fits[global_best_idx] << " | " << h_x_fits[global_best_idx] << std::endl;

    cudaFree(d_xs);
    cudaFree(d_local_best_xs);
    cudaFree(d_global_best_x);
    cudaFree(d_x_fits);
    cudaFree(d_local_best_fits);
    cudaFree(d_global_best_fit);
    delete[] h_xs;
    delete[] h_local_best_xs;
    delete[] h_global_best_x;
    delete[] x_fits;
    delete[] h_x_fits;
    delete[] h_local_best_fits;
    delete[] h_global_best_fit;
}


int main(){

    // test_mutex();
    // test_curand();
    // test_levy_function();
    // test_update_velocity();
    // test_update_position();
    // test_update_best();
    cudaDeviceSynchronize();
    return 0;
}

#endif