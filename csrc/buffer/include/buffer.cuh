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

template <typename ElemType>
class Buffer 
{
public:
    Buffer            () = delete;
    Buffer            (ssize_t nrow, ssize_t ncol, Device device);
    Buffer            (Buffer const  &other);
    Buffer            (Buffer       &&other) noexcept;
    Buffer & operator=(Buffer const  &other);
    Buffer & operator=(Buffer       &&other) noexcept;
    ~Buffer           ();

    void     set_value (ssize_t row, ssize_t col, ElemType val);
    ElemType get_value (ssize_t row, ssize_t col) const;
    ElemType operator()(ssize_t row, ssize_t col) const;

    ElemType *       data_ptr () const;
    ElemType const * cdata_ptr() const; 

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
    void fill (ElemType val);
    void clear();

    void copy_to_numpy(ndarray_t<ElemType> out) const;

private:
    ElemType  *m_buffer;
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
};