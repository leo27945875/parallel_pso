#pragma once

#include <vector>
#include <utility>
#include <pybind11/numpy.h>

namespace py = pybind11;

typedef std::pair<size_t, size_t>                                      shape_t;
typedef py::array_t<double, py::array::c_style | py::array::forcecast> ndarray_t;

enum class Device {
    CPU,
    GPU
};

class Buffer 
{
public:
    Buffer            () = delete                                       ;
    Buffer            (size_t nrow, size_t ncol, Device device)         ;
    Buffer            (Buffer const  &other)                            ;  // copy ctor
    Buffer            (Buffer       &&other)                    noexcept;  // move ctor
    Buffer & operator=(Buffer const  &other)                            ;  // copy assignment
    Buffer & operator=(Buffer       &&other)                    noexcept;  // move assignment
    ~Buffer           ();

    double set_value (size_t row, size_t col, double val)      ;
    double get_value (size_t row, size_t col)             const;
    double operator()(size_t row, size_t col)             const;

    double * data_ptr      ()                       const;
    shape_t  shape         ()                       const;
    size_t   nrow          ()                       const;
    size_t   ncol          ()                       const;
    size_t   num_elem      ()                       const;
    size_t   buffer_size   ()                       const;
    size_t   index_at      (size_t row, size_t col) const;
    Device   device        ()                       const;
    bool     is_same_shape (Buffer const &other)    const;
    bool     is_same_device(Buffer const &other)    const;
    void     copy_to_numpy (ndarray_t out)          const;

    void to   (Device device);
    void fill (double val)   ;
    void clear()             ;

private:
    double  *m_buffer;
    size_t   m_nrow;
    size_t   m_ncol;
    Device   m_device;

    void _release();
};