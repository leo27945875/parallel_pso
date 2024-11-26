#pragma once

#include <vector>
#include <utility>
#include <pybind11/numpy.h>

#include "utils.cuh"

namespace py = pybind11;

using shape_t = std::pair<ssize_t, ssize_t>;
template<typename T> using ndarray_t = py::array_t<T, py::array::c_style | py::array::forcecast>;


enum class Device {
    CPU,
    GPU
};

template <typename scalar_t>
class Buffer 
{
public:
    Buffer            () = delete;
    Buffer            (ssize_t nrow, ssize_t ncol = 1, Device device = Device::GPU);
    Buffer            (Buffer const  &other);
    Buffer            (Buffer       &&other) noexcept;
    Buffer & operator=(Buffer const  &other);
    Buffer & operator=(Buffer       &&other) noexcept;
    ~Buffer           ();

    void     set_value (ssize_t row, ssize_t col, scalar_t val);
    scalar_t get_value (ssize_t row, ssize_t col) const;
    scalar_t operator()(ssize_t row, ssize_t col) const;

    scalar_t *       data_ptr () const;
    scalar_t const * cdata_ptr() const; 

    Device      device        () const;
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
    void fill (scalar_t val);
    void clear();

    void copy_to_numpy(ndarray_t<scalar_t> &out) const;
    void copy_from_numpy(ndarray_t<scalar_t> const &src) const;

private:
    scalar_t  *m_buffer;
    ssize_t    m_nrow;
    ssize_t    m_ncol;
    Device     m_device;

    void _release();
};

template class Buffer<float>;
template class Buffer<double>;


class CURANDStates 
{
public:
    CURANDStates            () = delete;
    CURANDStates            (ssize_t size, unsigned long long seed);
    CURANDStates            (CURANDStates const  &other);
    CURANDStates            (CURANDStates       &&other) noexcept;
    CURANDStates & operator=(CURANDStates const  &other);
    CURANDStates & operator=(CURANDStates       &&other) noexcept;
    ~CURANDStates           ();

    cuda_rng_t *       data_ptr () const;
    cuda_rng_t const * cdata_ptr() const;

    ssize_t     num_elem   () const;
    ssize_t     buffer_size() const;
    std::string to_string  () const;

    void clear();

private:
    cuda_rng_t *m_buffer;
    ssize_t     m_size;

    void _release();
};