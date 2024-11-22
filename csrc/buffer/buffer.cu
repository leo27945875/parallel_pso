#include <algorithm>
#include <stdexcept>
#include <thrust/fill.h>

#include "buffer.cuh"


Buffer::Buffer(size_t nrow, size_t ncol, Device device)
    : m_nrow(nrow), m_ncol(ncol), m_device(device)
{
    switch (m_device)
    {
    case Device::CPU:
        m_buffer = new double[num_elem()];
        break;
    case Device::GPU:
        cudaMalloc(&m_buffer, buffer_size());
        break;
    }
}
Buffer::Buffer(Buffer const &other)
    : m_nrow(other.m_nrow), m_ncol(other.m_ncol), m_device(other.device())
{
    switch (m_device)
    {
    case Device::CPU:
        m_buffer = new double[num_elem()];
        memcpy(m_buffer, other.m_buffer, buffer_size());
        break;
    case Device::GPU:
        cudaMalloc(&m_buffer, buffer_size());
        cudaMemcpy(m_buffer, other.m_buffer, buffer_size(), cudaMemcpyDeviceToDevice);
        break;
    }
}
Buffer::Buffer(Buffer &&other) noexcept 
    : m_nrow(other.m_nrow), m_ncol(other.m_ncol), m_device(other.device())
{
    m_buffer       = other.m_buffer;
    other.m_buffer = nullptr;
    other.m_nrow   = 0;
    other.m_ncol   = 0;
}
Buffer & Buffer::operator=(Buffer const &other){
    if (!is_same_shape(other))
        throw std::runtime_error("Shapes does not match.");
    if (!is_same_device(other))
        throw std::runtime_error("Devices does not match.");
    switch (m_device)
    {
    case Device::CPU:
        memcpy(m_buffer, other.m_buffer, buffer_size());
        break;
    case Device::GPU:
        cudaMemcpy(m_buffer, other.m_buffer, buffer_size(), cudaMemcpyDeviceToDevice);
        break;
    }
}
Buffer & Buffer::operator=(Buffer &&other) noexcept {
    _release();
    m_buffer = other.m_buffer;
    m_nrow   = other.m_nrow;
    m_ncol   = other.m_ncol;
    m_device = other.m_device;
}
Buffer::~Buffer(){
    _release();
}

double Buffer::set_value(size_t row, size_t col, double val) {
    switch (m_device)
    {
    case Device::CPU:
        m_buffer[row * m_ncol + col] = val;
        break;
    case Device::GPU:
        cudaMemcpy(m_buffer + row * m_ncol + col, &val, sizeof(double), cudaMemcpyHostToDevice);
        break;
    }
}
double Buffer::get_value(size_t row, size_t col) const {
    double res;
    switch (m_device)
    {
    case Device::CPU:
        res = m_buffer[row * m_ncol + col];
        break;
    case Device::GPU:
        cudaMemcpy(&res, m_buffer + row * m_ncol + col, sizeof(double), cudaMemcpyDeviceToHost);
        break;
    }
    return res;
}
double Buffer::operator()(size_t row, size_t col) const {
    return get_value(row, col);
}

double * Buffer::data_ptr() const {
    return m_buffer;
}
shape_t Buffer::shape() const {
    return {m_nrow, m_ncol};
}
size_t Buffer::nrow() const {
    return m_nrow;
}
size_t Buffer::ncol() const {
    return m_ncol;
}
size_t Buffer::num_elem() const {
    return m_nrow * m_ncol;
}
size_t Buffer::buffer_size() const {
    return m_nrow * m_ncol * sizeof(double);
}
bool Buffer::is_same_shape(Buffer const &other) const {
    return m_nrow == other.nrow() && m_ncol == other.ncol();
}
bool Buffer::is_same_device(Buffer const &other) const {
    return m_device == other.device();
}
void Buffer::copy_to_numpy (ndarray_t out) const {
    // TODO
}

void Buffer::to(Device device){
    if (m_device == device)
        return;
    double *new_buffer;
    switch (m_device)
    {
    case Device::CPU:
        cudaMalloc(&new_buffer, buffer_size());
        cudaMemcpy(new_buffer, m_buffer, buffer_size(), cudaMemcpyHostToDevice);
        delete[] m_buffer;
        break;
    case Device::GPU:
        new_buffer = new double[num_elem()];
        cudaMemcpy(new_buffer, m_buffer, buffer_size(), cudaMemcpyDeviceToHost);
        cudaFree(m_buffer);
        break;
    }
    m_buffer = new_buffer;
}
void Buffer::fill(double val){
    switch (m_device)
    {
    case Device::CPU:
        std::fill_n(m_buffer, num_elem(), val);
        break;
    case Device::GPU:
        thrust::fill_n(m_buffer, num_elem(), val);
        break;
    }
}
void Buffer::clear(){
    _release();
    m_buffer = nullptr;
    m_nrow   = 0;
    m_ncol   = 0;
}

void Buffer::_release(){
    switch (m_device)
    {
    case Device::CPU:
        delete[] m_buffer;
        break;
    case Device::GPU:
        cudaFree(m_buffer);
        break;
    }
}