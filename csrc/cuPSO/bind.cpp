#include <pybind11/pybind11.h>
#include "utils.cuh"
#include "buffer.cuh"
#include "funcs.cuh"
#include "velocity.cuh"
#include "position.cuh"
#include "evolve.cuh"

namespace py = pybind11;


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
    Buffer const &xs, 
    Buffer       &out
){
    levy_function_cuda(
        xs.cdata_ptr(),
        out.data_ptr(),
        xs.nrow(),
        xs.ncol()
#if IS_CUDA_ALIGN_MALLOC
      , xs.pitch()
#endif
    );
}

void binded_update_velocities(
    Buffer const &xs,
    Buffer       &vs, 
    Buffer const &local_best_xs, 
    Buffer const &global_best_x,
    Buffer       &v_sum_pow2,
    scalar_t      w,
    scalar_t      c0,
    scalar_t      c1,
    scalar_t      v_max,
    CURANDStates &rng_states
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
#if IS_CUDA_ALIGN_MALLOC
        vs.pitch(),
        xs.pitch(),
        local_best_xs.pitch(),
        rng_states.pitch(),
#endif
        rng_states.data_ptr()
    );
}

void binded_update_positions(
    Buffer       &xs,
    Buffer const &vs,
    scalar_t      x_min,
    scalar_t      x_max
){
    update_positions_cuda(
        xs.data_ptr(),
        vs.cdata_ptr(),
        x_min,
        x_max,
        xs.nrow(),
        vs.ncol()
#if IS_CUDA_ALIGN_MALLOC
      , xs.pitch(),
        vs.pitch()
#endif
    );
}

ssize_t binded_update_bests(
    Buffer const &xs,
    Buffer const &x_fits,
    Buffer       &local_best_xs,
    Buffer       &local_best_fits,
    Buffer       &global_best_x,
    Buffer       &global_best_fit
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
#if IS_CUDA_ALIGN_MALLOC
      , xs.pitch(),
        local_best_xs.pitch()
#endif
    );
}


PYBIND11_MODULE(cuPSO, m){
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("GPU", Device::GPU);

    py::class_<Buffer>(m, "Buffer")
        .def(py::init<ssize_t, ssize_t, Device>(), py::arg("nrow"), py::arg("ncol") = 1, py::arg("device") = Device::GPU)
        .def("__getitem__"     , [](Buffer &self, std::pair<ssize_t, ssize_t> key)              { return self.get_value(key.first, key.second);      }, py::arg("key")                )
        .def("__setitem__"     , [](Buffer &self, std::pair<ssize_t, ssize_t> key, scalar_t val){ return self.set_value(key.first, key.second, val); }, py::arg("key"), py::arg("val"))
        .def("__repr__"        , &Buffer::to_string                           )
        .def("__str__"         , &Buffer::to_elem_string                      )
        .def("device"          , &Buffer::device                              )
        .def("shape"           , &Buffer::shape                               )
        .def("nrow"            , &Buffer::nrow                                )
        .def("ncol"            , &Buffer::ncol                                )
        .def("num_elem"        , &Buffer::num_elem                            )
        .def("buffer_size"     , &Buffer::buffer_size                         )
        .def("is_same_shape"   , &Buffer::is_same_shape    , py::arg("other") )
        .def("is_same_device"  , &Buffer::is_same_device   , py::arg("other") )
        .def("copy_to_numpy"   , &Buffer::copy_to_numpy    , py::arg("out")   )
        .def("copy_to_buffer"  , &Buffer::copy_to_buffer   , py::arg("out")   )
        .def("copy_from_numpy" , &Buffer::copy_from_numpy  , py::arg("src")   )
        .def("copy_from_buffer", &Buffer::copy_from_buffer , py::arg("src")   )
        .def("to"              , &Buffer::to               , py::arg("device"))
        .def("fill"            , &Buffer::fill             , py::arg("val")   )
        .def("show"            , &Buffer::show                                )
        .def("clear"           , &Buffer::clear                               );
    
    py::class_<CURANDStates>(m, "CURANDStates")
        .def(py::init<unsigned long long, ssize_t, ssize_t>(), py::arg("seed"), py::arg("nrow"), py::arg("ncol"))
        .def("__repr__"   , &CURANDStates::to_string  )
        .def("__str__"    , &CURANDStates::to_string  )
        .def("num_elem"   , &CURANDStates::num_elem   )
        .def("buffer_size", &CURANDStates::buffer_size)
        .def("clear"      , &CURANDStates::clear      );

    m.def("calc_fitness_val_npy" , &binded_calc_fitness_val_npy , py::arg("x")                 );
    m.def("calc_fitness_vals_npy", &binded_calc_fitness_vals_npy, py::arg("xs"), py::arg("out"));
    m.def("calc_fitness_vals"    , &binded_calc_fitness_vals    , py::arg("xs"), py::arg("out"));
    
    m.def("update_positions" , &binded_update_positions , py::arg("xs"), py::arg("vs"), py::arg("x_min"), py::arg("x_max"));
    m.def("update_bests"     , &binded_update_bests     , py::arg("xs"), py::arg("x_fits"), py::arg("local_best_xs"), py::arg("local_best_fits"), py::arg("global_best_x"), py::arg("global_best_fit"));
    m.def("update_velocities", &binded_update_velocities, py::arg("xs"), py::arg("vs"), py::arg("local_best_xs"), py::arg("global_best_x"), py::arg("v_sum_pow2"), py::arg("w"), py::arg("c0"), py::arg("c1"), py::arg("v_max"), py::arg("rng_states"));
}