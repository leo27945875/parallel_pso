#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <fstream>
#include <chrono>
#include <numeric> // For calculating mean and standard deviation

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>

#define IS_PARALLEL_GBEST 1

using namespace std;
namespace py = pybind11;
using array_t = py::array_t<double, py::array::c_style | py::array::forcecast>;


void set_num_threads(int n){
    omp_set_num_threads(n);
}

void calc_fitness_vals(array_t& array, array_t& result_array) {
    size_t n = array.shape(0);  // particle number
    size_t d = array.shape(1);  // particle dimensions
    double* ptr = array.mutable_data();
    double* result_ptr = result_array.mutable_data();

    auto w = [](double z) { return 1.0 + 0.25 * (z - 1.0); };

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        double result = pow(sin(M_PI * w(ptr[i * d + 0])), 2.0);
        for (size_t j = 0; j < d - 1; ++j) {
            result += pow((w(ptr[i * d + j]) - 1.0), 2.0) *
                    (1.0 + 10.0 * pow(sin(M_PI * w(ptr[i * d + j]) + 1.0), 2.0));
        }
        result += pow((w(ptr[i * d + d - 1]) - 1.0), 2.0) *
                (1.0 + pow(sin(2.0 * M_PI * w(ptr[i * d + d - 1])), 2.0));
        result_ptr[i] = result;
    }

}

void update_velocities_and_positions(
    array_t& positions, 
    array_t& velocities, 
    array_t& p_best_positions, 
    array_t& g_best_position, 
    double W, 
    double C0, 
    double C1, 
    double x_min, 
    double x_max, 
    double v_min, 
    double v_max
) {
    double* positions_ptr = positions.mutable_data();
    double* velocities_ptr = velocities.mutable_data();
    const double* p_best_positions_ptr = p_best_positions.data();
    const double* g_best_position_ptr = g_best_position.data();
    int dimensions = positions.shape(1);

    #pragma omp parallel for
    for (int i = 0; i < positions.shape(0); ++i) {

        unsigned int seed = time(NULL) + omp_get_thread_num();

        for (int j = 0; j < dimensions; ++j) {

            double r0 = (double)rand_r(&seed) / RAND_MAX;
            double r1 = (double)rand_r(&seed) / RAND_MAX;

            velocities_ptr[i * dimensions + j] = W * velocities_ptr[i * dimensions + j] +
                                                 C0 * r0 * (p_best_positions_ptr[i * dimensions + j] - positions_ptr[i * dimensions + j]) +
                                                 C1 * r1 * (g_best_position_ptr[j] - positions_ptr[i * dimensions + j]);

            if (velocities_ptr[i * dimensions + j] < v_min) velocities_ptr[i * dimensions + j] = v_min;
            if (velocities_ptr[i * dimensions + j] > v_max) velocities_ptr[i * dimensions + j] = v_max;

            positions_ptr[i * dimensions + j] += velocities_ptr[i * dimensions + j];

            if (positions_ptr[i * dimensions + j] < x_min) positions_ptr[i * dimensions + j] = x_min;
            if (positions_ptr[i * dimensions + j] > x_max) positions_ptr[i * dimensions + j] = x_max;

        }
    }

}

#if not IS_PARALLEL_GBEST
void update_best_values(
    array_t& positions,
    array_t& p_best_positions,
    array_t& fitness_values,
    array_t& p_best_values,
    array_t& g_best_position,
    array_t& g_best_value
) {

    double* positions_ptr = positions.mutable_data();
    double* p_best_positions_ptr = p_best_positions.mutable_data();
    double* fitness_values_ptr = fitness_values.mutable_data();
    double* p_best_values_ptr = p_best_values.mutable_data();
    double* g_best_position_ptr = g_best_position.mutable_data();
    double* g_best_value_ptr = g_best_value.mutable_data();

    int dimensions = positions.shape(1);

    for (int i = 0; i < positions.shape(0); ++i) {
        if (fitness_values_ptr[i] < p_best_values_ptr[i]) {
            p_best_values_ptr[i] = fitness_values_ptr[i];

            for (size_t j = 0; j < dimensions; ++j) {
                p_best_positions_ptr[i * dimensions + j] = positions_ptr[i * dimensions + j];
            }

            if (fitness_values_ptr[i] < g_best_value_ptr[0]) {
                g_best_value_ptr[0] = fitness_values_ptr[i];
                    
                for (size_t j = 0; j < dimensions; ++j) {
                    g_best_position_ptr[j] = positions_ptr[i * dimensions + j];
                }
            }

        }
    }

}
#else
void update_best_values(
    array_t& positions,
    array_t& p_best_positions,
    array_t& fitness_values,
    array_t& p_best_values,
    array_t& g_best_position,
    array_t& g_best_value
) {
    double* positions_ptr = positions.mutable_data();
    double* p_best_positions_ptr = p_best_positions.mutable_data();
    double* fitness_values_ptr = fitness_values.mutable_data();
    double* p_best_values_ptr = p_best_values.mutable_data();
    double* g_best_position_ptr = g_best_position.mutable_data();
    double* g_best_value_ptr = g_best_value.mutable_data();

    int dimensions = positions.shape(1);
    int num_particles = positions.shape(0);

    // 設定線程數
    int num_threads = omp_get_max_threads();

    // 初始化數據
    double* local_g_best_value_ptr = new double [num_threads];
    double* local_g_best_position_ptr = new double [num_threads * dimensions];
    for (int t = 0; t < num_threads; ++t)
        local_g_best_value_ptr[t] = std::numeric_limits<double>::max();

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num(); // 獲取線程 ID
        for (int i = thread_id; i < num_particles; i += num_threads) {
            // 更新每個粒子的個體最佳值
            if (fitness_values_ptr[i] < p_best_values_ptr[i]) {
                p_best_values_ptr[i] = fitness_values_ptr[i];
                for (int j = 0; j < dimensions; ++j) {
                    p_best_positions_ptr[i * dimensions + j] = positions_ptr[i * dimensions + j];
                }
            }

            // 更新 local_g_best_value 和 local_g_best_position
            if (fitness_values_ptr[i] < local_g_best_value_ptr[thread_id]) {
                local_g_best_value_ptr[thread_id] = fitness_values_ptr[i];
                for (int j = 0; j < dimensions; ++j) {
                    local_g_best_position_ptr[thread_id * dimensions + j] = positions_ptr[i * dimensions + j];
                }
            }
        }
    }

    // 主執行緒合併 local_g_best_value 和 local_g_best_position
    for (int t = 0; t < num_threads; ++t) {
        if (local_g_best_value_ptr[t] < g_best_value_ptr[0]) {
            g_best_value_ptr[0] = local_g_best_value_ptr[t];
            for (int j = 0; j < dimensions; ++j) {
                g_best_position_ptr[j] = local_g_best_position_ptr[t * dimensions + j];
            }
        }
    }
}
#endif

PYBIND11_MODULE(ompPSO, m) {
    m.def("set_num_threads"                , &set_num_threads);
    m.def("calc_fitness_vals"              , &calc_fitness_vals              , py::arg("array"), py::arg("result_array"));
    m.def("update_velocities_and_positions", &update_velocities_and_positions, py::arg("positions"), py::arg("velocities"), py::arg("p_best_positions"), py::arg("g_best_position"), py::arg("W"), py::arg("C0"), py::arg("C1"), py::arg("x_min"), py::arg("x_max"), py::arg("v_min"), py::arg("v_max"));
    m.def("update_best_values"             , &update_best_values             , py::arg("positions"), py::arg("p_best_positions"), py::arg("fitness_values"), py::arg("p_best_values"), py::arg("g_best_position"), py::arg("g_best_value"));
}
