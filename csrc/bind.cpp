#include <pybind11/pybind11.h>

#include "buffer.cuh"
#include "funcs.cuh"
#include "velocity.cuh"
#include "position.cuh"
#include "evolve.cuh"

namespace py = pybind11;

using scalar_t = double;
using FloatBuffer = Buffer<scalar_t>;


scalar_t binded_calc_fitness_val_npy(
    ndarray_t<scalar_t> const &x
){
    return levy(x.data(), x.shape(0));
}
void binded_calc_fitness_vals_npy(
    ndarray_t<scalar_t> const &xs,
    ndarray_t<scalar_t>       &out
){
    levy_function_cpu(
        xs.data(),
        out.mutable_data(),
        xs.shape(0),
        xs.shape(1)
    );
}
void binded_calc_fitness_vals(
    FloatBuffer const &xs, 
    FloatBuffer       &out
){
    levy_function_cuda(
        xs.cdata_ptr(),
        out.data_ptr(),
        xs.nrow(),
        xs.ncol()
    );
}

void binded_update_velocities(
    FloatBuffer const &xs,
    FloatBuffer       &vs, 
    FloatBuffer const &local_best_xs, 
    FloatBuffer const &global_best_x,
    FloatBuffer       &v_sum_pow2,
    double             w,
    double             c0,
    double             c1,
    double             v_max,
    CURANDStates      &rng_states
){
    update_velocities_cuda(
        vs.data_ptr(),
        xs.cdata_ptr(),
        local_best_xs.cdata_ptr(),
        global_best_x.cdata_ptr(),
        v_sum_pow2.data_ptr(),
        w,
        c0,
        c1,
        v_max,
        xs.nrow(),
        xs.ncol(),
        rng_states.data_ptr()
    );
}

void binded_update_positions(
    FloatBuffer       &xs,
    FloatBuffer const &vs,
    double             x_min,
    double             x_max
){
    update_positions_cuda(
        xs.data_ptr(),
        vs.cdata_ptr(),
        x_min,
        x_max,
        xs.nrow(),
        vs.ncol()
    );
}

ssize_t binded_update_bests(
    FloatBuffer const &xs,
    FloatBuffer const &x_fits,
    FloatBuffer       &local_best_xs,
    FloatBuffer       &local_best_fits,
    FloatBuffer       &global_best_x,
    FloatBuffer       &global_best_fit
){
    return update_bests_cuda(
        xs.cdata_ptr(),
        x_fits.cdata_ptr(),
        local_best_xs.data_ptr(),
        local_best_fits.data_ptr(),
        global_best_x.data_ptr(),
        global_best_fit.data_ptr(),
        xs.nrow(),
        xs.ncol()
    );
}


PYBIND11_MODULE(cuPSO, m){
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("GPU", Device::GPU);

    py::class_<FloatBuffer>(m, "Buffer")
        .def(py::init<ssize_t, ssize_t, Device>(), py::arg("nrow"), py::arg("ncol") = 1, py::arg("device") = Device::GPU)
        .def("__getitem__"    , [](FloatBuffer &self, std::pair<ssize_t, ssize_t> key)            { return self.get_value(key.first, key.second);      }, py::arg("key")                )
        .def("__setitem__"    , [](FloatBuffer &self, std::pair<ssize_t, ssize_t> key, double val){ return self.set_value(key.first, key.second, val); }, py::arg("key"), py::arg("val"))
        .def("__repr__"       , &FloatBuffer::to_string                          )
        .def("__str__"        , &FloatBuffer::to_string                          )
        .def("device"         , &FloatBuffer::device                             )
        .def("shape"          , &FloatBuffer::shape                              )
        .def("nrow"           , &FloatBuffer::nrow                               )
        .def("ncol"           , &FloatBuffer::ncol                               )
        .def("num_elem"       , &FloatBuffer::num_elem                           )
        .def("buffer_size"    , &FloatBuffer::buffer_size                        )
        .def("is_same_shape"  , &FloatBuffer::is_same_shape   , py::arg("other") )
        .def("is_same_device" , &FloatBuffer::is_same_device  , py::arg("other") )
        .def("copy_to_numpy"  , &FloatBuffer::copy_to_numpy   , py::arg("out")   )
        .def("copy_from_numpy", &FloatBuffer::copy_from_numpy , py::arg("src")   )
        .def("to"             , &FloatBuffer::to              , py::arg("device"))
        .def("fill"           , &FloatBuffer::fill            , py::arg("val")   )
        .def("show"           , &FloatBuffer::show                               )
        .def("clear"          , &FloatBuffer::clear                              );
    
    py::class_<CURANDStates>(m, "CURANDStates")
        .def(py::init<ssize_t, unsigned long long>(), py::arg("size"), py::arg("seed"))
        .def("__repr__"   , &CURANDStates::to_string  )
        .def("__str__"    , &CURANDStates::to_string  )
        .def("num_elem"   , &CURANDStates::num_elem   )
        .def("buffer_size", &CURANDStates::buffer_size)
        .def("clear"      , &CURANDStates::clear      );

    m.def("calc_fitness_val_npy" , &binded_calc_fitness_val_npy , py::arg("x")                 );
    m.def("calc_fitness_vals_npy", &binded_calc_fitness_vals_npy, py::arg("xs"), py::arg("out"));
    m.def("calc_fitness_vals"    , &binded_calc_fitness_vals    , py::arg("xs"), py::arg("out"));
    
    m.def("update_velocities", &binded_update_velocities, py::arg("xs"), py::arg("vs")    , py::arg("local_best_xs"), py::arg("global_best_x"), py::arg("v_sum_pow2"), py::arg("w"), py::arg("c0"), py::arg("c1"), py::arg("v_max"), py::arg("rng_states"));
    m.def("update_positions" , &binded_update_positions , py::arg("xs"), py::arg("vs")    , py::arg("x_min")        , py::arg("x_max")                                                                                                                    );
    m.def("update_bests"     , &binded_update_bests     , py::arg("xs"), py::arg("x_fits"), py::arg("local_best_xs"), py::arg("local_best_fits"), py::arg("global_best_x"), py::arg("global_best_fit")                                                    );
}