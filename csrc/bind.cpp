#include <pybind11/pybind11.h>

#include "buffer.cuh"

namespace py = pybind11;


PYBIND11_MODULE(cuPSO, m){
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("GPU", Device::GPU)
        .export_values();

    py::class_<Buffer>(m, "Buffer")
        .def(py::init<ssize_t, ssize_t, Device>())
        .def("__getitem__"   , [](Buffer &self, std::pair<ssize_t, ssize_t> key)            { return self.get_value(key.first, key.second);      })
        .def("__setitem__"   , [](Buffer &self, std::pair<ssize_t, ssize_t> key, double val){ return self.set_value(key.first, key.second, val); })
        .def("__repr__"      , &Buffer::to_string     )
        .def("__str__"       , &Buffer::to_string     )
        .def("device"        , &Buffer::device        )
        .def("shape"         , &Buffer::shape         )
        .def("nrow"          , &Buffer::nrow          )
        .def("ncol"          , &Buffer::ncol          )
        .def("num_elem"      , &Buffer::num_elem      )
        .def("buffer_size"   , &Buffer::buffer_size   )
        .def("is_same_shape" , &Buffer::is_same_shape )
        .def("is_same_device", &Buffer::is_same_device)
        .def("copy_to_numpy" , &Buffer::copy_to_numpy )
        .def("to"            , &Buffer::to            )
        .def("fill"          , &Buffer::fill          )
        .def("clear"         , &Buffer::clear         );
}