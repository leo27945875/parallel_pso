#pragma once

#include <vector>
#include <utility>
#include <pybind11/numpy.h>

namespace py = pybind11;

typedef std::pair<ssize_t, ssize_t>                                    shape_t;
typedef py::array_t<double, py::array::c_style | py::array::forcecast> ndarray_t;

enum class Device {
    CPU,
    GPU
};

class Buffer 
{
public:
    Buffer            () = delete;
    Buffer            (ssize_t nrow, ssize_t ncol, Device device);
    Buffer            (Buffer const  &other);                      // copy ctor
    Buffer            (Buffer       &&other) noexcept;             // move ctor
    Buffer & operator=(Buffer const  &other);                      // copy assignment
    Buffer & operator=(Buffer       &&other) noexcept;             // move assignment
    ~Buffer           ();

    void   set_value (ssize_t row, ssize_t col, double val);
    double get_value (ssize_t row, ssize_t col) const;
    double operator()(ssize_t row, ssize_t col) const;

    Device      device        () const;
    double *    data_ptr      () const;
    shape_t     shape         () const;
    ssize_t     nrow          () const;
    ssize_t     ncol          () const;
    ssize_t     num_elem      () const;
    ssize_t     buffer_size   () const;
    ssize_t     index_at      (ssize_t row, ssize_t col) const;
    bool        is_same_shape (Buffer const &other) const;
    bool        is_same_device(Buffer const &other) const;
    std::string to_string     () const;

    void to   (Device device);
    void fill (double val);
    void clear();

    void copy_to_numpy(ndarray_t out) const;

private:
    double  *m_buffer = nullptr;
    ssize_t  m_nrow;
    ssize_t  m_ncol;
    Device   m_device;

    void _release();
};