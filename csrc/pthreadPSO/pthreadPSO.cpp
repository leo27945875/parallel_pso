#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <fstream>
#include <chrono>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pthread.h>

#define IS_PARALLEL_GBEST 1

using namespace std;
namespace py = pybind11;
using array_t = py::array_t<double, py::array::c_style | py::array::forcecast>;


struct ThreadData {
    size_t  start;
    size_t  end;
    double* array;
    double* result_array;
    size_t  d;
};
struct VelocityThreadData {
    size_t        start;
    size_t        end;
    double*       positions;
    double*       velocities;
    const double* p_best_positions;
    const double* g_best_position;
    int           dimensions;
    double        W;
    double        C0;
    double        C1;
    double        x_min;
    double        x_max;
    double        v_min;
    double        v_max;
    int           thread_id;
};
struct BestValuesThreadData {
    size_t              start;
    size_t              end;
    double*             positions;
    double*             p_best_positions;
    double*             fitness_values;
    double*             p_best_values;
    double*             g_best_position;
    double*             g_best_value;
    int                 dimensions;
    int                 thread_id;
    double              local_g_best_value;
    std::vector<double> local_g_best_position;
};


static size_t num_threads = 1;


void set_num_threads(size_t n){
    num_threads = n;
}

void* calc_fitness_vals_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    auto w = [](double z) { return 1.0 + 0.25 * (z - 1.0); };

    for (size_t i = data->start; i < data->end; ++i) {
        double result = pow(sin(M_PI * w(data->array[i * data->d])), 2.0);

        for (size_t j = 0; j < data->d - 1; ++j) {
            result += pow((w(data->array[i * data->d + j]) - 1.0), 2.0) *
                      (1.0 + 10.0 * pow(sin(M_PI * w(data->array[i * data->d + j]) + 1.0), 2.0));
        }

        result += pow((w(data->array[i * data->d + data->d - 1]) - 1.0), 2.0) *
                  (1.0 + pow(sin(2.0 * M_PI * w(data->array[i * data->d + data->d - 1])), 2.0));

        data->result_array[i] = result;
    }
    pthread_exit(nullptr);
}
void calc_fitness_vals(array_t& array, array_t& result_array) {
    size_t n = array.shape(0);
    size_t d = array.shape(1);
    double* ptr = array.mutable_data();
    double* result_ptr = result_array.mutable_data();

    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    size_t chunk_size = n / num_threads;

    for (size_t t = 0; t < num_threads; ++t) {
        thread_data[t].start = t * chunk_size;
        thread_data[t].end = (t == num_threads - 1) ? n : (t + 1) * chunk_size;
        thread_data[t].array = ptr;
        thread_data[t].result_array = result_ptr;
        thread_data[t].d = d;

        pthread_create(&threads[t], nullptr, calc_fitness_vals_thread, (void*)&thread_data[t]);
    }
    for (size_t t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], nullptr);
    }
}

void* update_velocity_thread(void* arg) {
    VelocityThreadData* data = (VelocityThreadData*)arg;
    for (size_t i = data->start; i < data->end; ++i) {

        unsigned int seed = time(NULL) + data->thread_id;

        for (int j = 0; j < data->dimensions; ++j) {

            double r1 = (double)rand_r(&seed) / RAND_MAX;
            double r2 = (double)rand_r(&seed) / RAND_MAX;

            double new_velocity = data->W * data->velocities[i * data->dimensions + j] +
                                  data->C0 * r1 * (data->p_best_positions[i * data->dimensions + j] - data->positions[i * data->dimensions + j]) +
                                  data->C1 * r2 * (data->g_best_position[j] - data->positions[i * data->dimensions + j]);

            if (new_velocity < data->v_min) new_velocity = data->v_min;
            if (new_velocity > data->v_max) new_velocity = data->v_max;

            data->velocities[i * data->dimensions + j] = new_velocity;

            double new_position = data->positions[i * data->dimensions + j] + new_velocity;

            if (new_position < data->x_min) new_position = data->x_min;
            if (new_position > data->x_max) new_position = data->x_max;

            data->positions[i * data->dimensions + j] = new_position;
        }
    }
    pthread_exit(nullptr);
}
void update_velocities_and_positions(array_t& positions, array_t& velocities, array_t& p_best_positions, array_t& g_best_position, double W, double C0, double C1, double x_min, double x_max, double v_min, double v_max) {
    int num_particles = positions.shape(0);
    int dimensions = positions.shape(1);

    double* positions_ptr = positions.mutable_data();
    double* velocities_ptr = velocities.mutable_data();
    const double* p_best_positions_ptr = p_best_positions.data();
    const double* g_best_position_ptr = g_best_position.data();

    pthread_t threads[num_threads];
    VelocityThreadData thread_data[num_threads];

    size_t chunk_size = num_particles / num_threads;

    for (size_t t = 0; t < num_threads; ++t) {
        thread_data[t].start = t * chunk_size;
        thread_data[t].end = (t == num_threads - 1) ? num_particles : (t + 1) * chunk_size;
        thread_data[t].positions = positions_ptr;
        thread_data[t].velocities = velocities_ptr;
        thread_data[t].p_best_positions = p_best_positions_ptr;
        thread_data[t].g_best_position = g_best_position_ptr;
        thread_data[t].dimensions = dimensions;
        thread_data[t].W = W;
        thread_data[t].C0 = C0;
        thread_data[t].C1 = C1;
        thread_data[t].x_min = x_min;
        thread_data[t].x_max = x_max;
        thread_data[t].v_min = v_min;
        thread_data[t].v_max = v_max;
        thread_data[t].thread_id = t;

        pthread_create(&threads[t], nullptr, update_velocity_thread, (void*)&thread_data[t]);
    }

    for (size_t t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], nullptr);
    }
}

void* update_best_values_thread(void* arg) {
    BestValuesThreadData* data = (BestValuesThreadData*)arg;
    data->local_g_best_value = std::numeric_limits<double>::max();
    data->local_g_best_position.resize(data->dimensions, 0.0);

    for (size_t i = data->start; i < data->end; ++i) {
        if (data->fitness_values[i] < data->p_best_values[i]) {
            data->p_best_values[i] = data->fitness_values[i];
            for (int j = 0; j < data->dimensions; ++j) {
                data->p_best_positions[i * data->dimensions + j] = data->positions[i * data->dimensions + j];
            }
        }
        if (data->fitness_values[i] < data->local_g_best_value) {
            data->local_g_best_value = data->fitness_values[i];

            for (int j = 0; j < data->dimensions; ++j) {
                data->local_g_best_position[j] = data->positions[i * data->dimensions + j];
            }
        }
    }
    pthread_exit(nullptr);
}
void update_best_values(
    array_t& positions,
    array_t& p_best_positions,
    array_t& fitness_values,
    array_t& p_best_values,
    array_t& g_best_position,
    array_t& g_best_value
) {
    int num_particles = positions.shape(0);
    int dimensions = positions.shape(1);

    double* positions_ptr = positions.mutable_data();
    double* p_best_positions_ptr = p_best_positions.mutable_data();
    double* fitness_values_ptr = fitness_values.mutable_data();
    double* p_best_values_ptr = p_best_values.mutable_data();
    double* g_best_position_ptr = g_best_position.mutable_data();
    double* g_best_value_ptr = g_best_value.mutable_data();

    pthread_t threads[num_threads];
    BestValuesThreadData thread_data[num_threads];
    size_t chunk_size = num_particles / num_threads;

    for (size_t t = 0; t < num_threads; ++t) {
        thread_data[t].start = t * chunk_size;
        thread_data[t].end = (t == num_threads - 1) ? num_particles : (t + 1) * chunk_size;
        thread_data[t].positions = positions_ptr;
        thread_data[t].p_best_positions = p_best_positions_ptr;
        thread_data[t].fitness_values = fitness_values_ptr;
        thread_data[t].p_best_values = p_best_values_ptr;
        thread_data[t].g_best_position = g_best_position_ptr;
        thread_data[t].g_best_value = g_best_value_ptr;
        thread_data[t].dimensions = dimensions;
        thread_data[t].thread_id = t;
        pthread_create(&threads[t], nullptr, update_best_values_thread, (void*)&thread_data[t]);
    }
    for (size_t t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], nullptr);
    }
    for (size_t t = 0; t < num_threads; ++t) {
        if (thread_data[t].local_g_best_value < g_best_value_ptr[0]) {
            g_best_value_ptr[0] = thread_data[t].local_g_best_value;

            for (int j = 0; j < dimensions; ++j) {
                g_best_position_ptr[j] = thread_data[t].local_g_best_position[j];
            }
        }
    }
}

PYBIND11_MODULE(pthreadPSO, m){
    m.def("set_num_threads"                , &set_num_threads);
    m.def("calc_fitness_vals"              , &calc_fitness_vals              , py::arg("array"), py::arg("result_array"));
    m.def("update_velocities_and_positions", &update_velocities_and_positions, py::arg("positions"), py::arg("velocities"), py::arg("p_best_positions"), py::arg("g_best_position"), py::arg("W"), py::arg("C0"), py::arg("C1"), py::arg("x_min"), py::arg("x_max"), py::arg("v_min"), py::arg("v_max"));
    m.def("update_best_values"             , &update_best_values             , py::arg("positions"), py::arg("p_best_positions"), py::arg("fitness_values"), py::arg("p_best_values"), py::arg("g_best_position"), py::arg("g_best_value"));
}
