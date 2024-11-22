#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>

#include "buffer.cuh"
#include "utils.cuh"


Buffer::Buffer(ssize_t nrow, ssize_t ncol, Device device)
    : m_nrow(nrow), m_ncol(ncol), m_device(device)
{
    switch (m_device)
    {
    case Device::CPU:
        m_buffer = new double[num_elem()];
        break;
    case Device::GPU:
        cudaMalloc(&m_buffer, buffer_size()); 
        cudaCheckErrors("Failed to allocate GPU buffer.");
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
    return *this;
}
Buffer & Buffer::operator=(Buffer &&other) noexcept {
    _release();
    m_buffer = other.m_buffer;
    m_nrow   = other.m_nrow;
    m_ncol   = other.m_ncol;
    m_device = other.m_device;
    return *this;
}
Buffer::~Buffer(){
    _release();
}

void Buffer::set_value(ssize_t row, ssize_t col, double val) {
    switch (m_device)
    {
    case Device::CPU:
        m_buffer[index_at(row, col)] = val;
        break;
    case Device::GPU:
        cudaMemcpy(m_buffer + index_at(row, col), &val, sizeof(double), cudaMemcpyHostToDevice);
        break;
    }
}
double Buffer::get_value(ssize_t row, ssize_t col) const {
    double res;
    switch (m_device)
    {
    case Device::CPU:
        res = m_buffer[index_at(row, col)];
        break;
    case Device::GPU:
        cudaMemcpy(&res, m_buffer + index_at(row, col), sizeof(double), cudaMemcpyDeviceToHost);
        break;
    }
    return res;
}
double Buffer::operator()(ssize_t row, ssize_t col) const {
    return get_value(row, col);
}

Device Buffer::device() const {
    return m_device;
}
double * Buffer::data_ptr() const {
    return m_buffer;
}
shape_t Buffer::shape() const {
    return {m_nrow, m_ncol};
}
ssize_t Buffer::nrow() const {
    return m_nrow;
}
ssize_t Buffer::ncol() const {
    return m_ncol;
}
ssize_t Buffer::num_elem() const {
    return m_nrow * m_ncol;
}
ssize_t Buffer::buffer_size() const {
    return m_nrow * m_ncol * sizeof(double);
}
ssize_t Buffer::index_at(ssize_t row, ssize_t col) const {
    return row * m_ncol + col;
}
bool Buffer::is_same_shape(Buffer const &other) const {
    return m_nrow == other.nrow() && m_ncol == other.ncol();
}
bool Buffer::is_same_device(Buffer const &other) const {
    return m_device == other.device();
}
std::string Buffer::to_string() const {
    std::stringstream ss;
    ss << "<Buffer shape=(" << nrow() << ", " << ncol() << ") device=" << ((m_device == Device::CPU)? "CPU": "GPU") << " @" << (uintptr_t)this << ">";
    return ss.str();
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
    m_device = device;
}
void Buffer::fill(double val){
    switch (m_device)
    {
    case Device::CPU:
        std::fill_n(m_buffer, num_elem(), val);
        break;
    case Device::GPU:
        thrust::device_ptr<double> dev_ptr(m_buffer);
        thrust::fill_n(dev_ptr, num_elem(), val); 
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
        cudaCheckErrors("Failed to free GPU buffer.");
        break;
    }
}

void Buffer::copy_to_numpy(ndarray_t out) const {
    ssize_t buf_buffer_size = buffer_size();
    ssize_t npy_buffer_size = out.nbytes();

    if (npy_buffer_size != buf_buffer_size)
        throw std::runtime_error("Size of numpy array does not match this buffer.");
    
    switch (m_device)
    {
    case Device::CPU:
        memcpy(out.mutable_data(), m_buffer, buf_buffer_size);
        break;
    case Device::GPU:
        cudaMemcpy(out.mutable_data(), m_buffer, buf_buffer_size, cudaMemcpyDeviceToHost);
        cudaCheckErrors("Failed to copy data from GPU buffer.");
        break;
    }
}