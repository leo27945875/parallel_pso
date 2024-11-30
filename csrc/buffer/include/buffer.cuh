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
    ssize_t     padded_size   () const;
    ssize_t     pitch         () const;
    ssize_t     default_pitch () const;
    bool        is_same_shape (Buffer const &other) const;
    bool        is_same_device(Buffer const &other) const;
    std::string to_string     () const;
    std::string to_elem_string() const;
    void        show          () const;
    void        to            (Device device);
    void        fill          (scalar_t val);
    void        clear         ();

    void copy_to_numpy   (ndarray_t<scalar_t>       &out) const;
    void copy_from_numpy (ndarray_t<scalar_t> const &src);
    void copy_to_buffer  (Buffer                    &out) const;
    void copy_from_buffer(Buffer              const &src);

private:
    scalar_t  *m_buffer;
    ssize_t    m_nrow;
    ssize_t    m_ncol;
    ssize_t    m_pitch;
    Device     m_device;

    scalar_t * _ptr_at (ssize_t row, ssize_t col) const;
    void       _release();
};


class CURANDStates 
{
public:
    CURANDStates            () = delete;
    CURANDStates            (unsigned long long seed, ssize_t nrow, ssize_t ncol = 1L);
    CURANDStates            (CURANDStates const  &other);
    CURANDStates            (CURANDStates       &&other) noexcept;
    CURANDStates & operator=(CURANDStates const  &other);
    CURANDStates & operator=(CURANDStates       &&other) noexcept;
    ~CURANDStates           ();

    cuda_rng_t *       data_ptr () const;
    cuda_rng_t const * cdata_ptr() const;

    ssize_t     num_elem   () const;
    ssize_t     buffer_size() const;
    ssize_t     pitch      () const;
    std::string to_string  () const;

    void clear();

private:
    cuda_rng_t *m_buffer;
    ssize_t     m_pitch;
    ssize_t     m_nrow;
    ssize_t     m_ncol;

    void _release();
};
