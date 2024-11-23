#include <pybind11/pybind11.h>

#include "buffer.cuh"
// #include "funcs.cuh"
// #include "velocity.cuh"
// #include "position.cuh"
// #include "evolve.cuh"

namespace py = pybind11;

using FloatBuffer = Buffer<double>;


PYBIND11_MODULE(cuPSO, m){
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("GPU", Device::GPU);

    py::class_<FloatBuffer>(m, "Buffer")
        .def(py::init<ssize_t, ssize_t, Device>(), py::arg("nrow"), py::arg("ncol"), py::arg("device"))
        .def("__getitem__"   , [](FloatBuffer &self, std::pair<ssize_t, ssize_t> key)            { return self.get_value(key.first, key.second);      }, py::arg("key")                )
        .def("__setitem__"   , [](FloatBuffer &self, std::pair<ssize_t, ssize_t> key, double val){ return self.set_value(key.first, key.second, val); }, py::arg("key"), py::arg("val"))
        .def("__repr__"      , &FloatBuffer::to_string                        )
        .def("__str__"       , &FloatBuffer::to_string                        )
        .def("device"        , &FloatBuffer::device                           )
        .def("shape"         , &FloatBuffer::shape                            )
        .def("nrow"          , &FloatBuffer::nrow                             )
        .def("ncol"          , &FloatBuffer::ncol                             )
        .def("num_elem"      , &FloatBuffer::num_elem                         )
        .def("buffer_size"   , &FloatBuffer::buffer_size                      )
        .def("is_same_shape" , &FloatBuffer::is_same_shape , py::arg("other") )
        .def("is_same_device", &FloatBuffer::is_same_device, py::arg("other") )
        .def("copy_to_numpy" , &FloatBuffer::copy_to_numpy , py::arg("out")   )
        .def("to"            , &FloatBuffer::to            , py::arg("device"))
        .def("fill"          , &FloatBuffer::fill          , py::arg("val")   )
        .def("clear"         , &FloatBuffer::clear                            );
    
    py::class_<CURANDStates>(m, "CURANDStates")
        .def(py::init<ssize_t, unsigned long long>(), py::arg("size"), py::arg("seed"))
        .def("__repr__"   , &CURANDStates::to_string  )
        .def("__str__"    , &CURANDStates::to_string  )
        .def("num_elem"   , &CURANDStates::num_elem   )
        .def("buffer_size", &CURANDStates::buffer_size)
        .def("clear"      , &CURANDStates::clear      );
}