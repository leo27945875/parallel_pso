#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>

#include "buffer.cuh"


// start Buffer
template <typename ElemType>
Buffer<ElemType>::Buffer(ssize_t nrow, ssize_t ncol, Device device)
    : m_nrow(nrow), m_ncol(ncol), m_device(device)
{
    if (nrow == 0 || ncol == 0){
        m_buffer = nullptr;
        return;
    }
    switch (m_device)
    {
    case Device::CPU:
        m_buffer = new ElemType[num_elem()];
        break;
    case Device::GPU:
        cudaMalloc(&m_buffer, buffer_size()); 
        cudaCheckErrors("Failed to allocate GPU buffer.");
        break;
    }
}
template <typename ElemType>
Buffer<ElemType>::Buffer(Buffer<ElemType> const &other)
    : m_nrow(other.m_nrow), m_ncol(other.m_ncol), m_device(other.m_device)
{
    switch (m_device)
    {
    case Device::CPU:
        m_buffer = new ElemType[num_elem()];
        memcpy(m_buffer, other.m_buffer, buffer_size());
        break;
    case Device::GPU:
        cudaMalloc(&m_buffer, buffer_size());
        cudaMemcpy(m_buffer, other.m_buffer, buffer_size(), cudaMemcpyDeviceToDevice);
        break;
    }
}
template <typename ElemType>
Buffer<ElemType>::Buffer(Buffer<ElemType> &&other) noexcept 
    : m_nrow(other.m_nrow), m_ncol(other.m_ncol), m_device(other.m_device), m_buffer(other.m_buffer)
{
    other.m_buffer = nullptr;
    other.m_nrow   = 0;
    other.m_ncol   = 0;
}
template <typename ElemType>
Buffer<ElemType> & Buffer<ElemType>::operator=(Buffer<ElemType> const &other){
    if (!is_same_shape(other))
        throw std::runtime_error("Shapes do not match.");
    if (!is_same_device(other))
        throw std::runtime_error("Devices do not match.");
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
template <typename ElemType>
Buffer<ElemType> & Buffer<ElemType>::operator=(Buffer<ElemType> &&other) noexcept {
    _release();
    m_buffer       = other.m_buffer;
    m_nrow         = other.m_nrow;
    m_ncol         = other.m_ncol;
    m_device       = other.m_device;
    other.m_buffer = nullptr;
    other.m_nrow   = 0;
    other.m_ncol   = 0;
    return *this;
}
template <typename ElemType>
Buffer<ElemType>::~Buffer(){
    _release();
}

template <typename ElemType>
void Buffer<ElemType>::set_value(ssize_t row, ssize_t col, ElemType val) {
    switch (m_device)
    {
    case Device::CPU:
        m_buffer[index_at(row, col)] = val;
        break;
    case Device::GPU:
        cudaMemcpy(m_buffer + index_at(row, col), &val, sizeof(ElemType), cudaMemcpyHostToDevice);
        break;
    }
}
template <typename ElemType>
ElemType Buffer<ElemType>::get_value(ssize_t row, ssize_t col) const {
    ElemType res;
    switch (m_device)
    {
    case Device::CPU:
        res = m_buffer[index_at(row, col)];
        break;
    case Device::GPU:
        cudaMemcpy(&res, m_buffer + index_at(row, col), sizeof(ElemType), cudaMemcpyDeviceToHost);
        break;
    }
    return res;
}
template <typename ElemType>
ElemType Buffer<ElemType>::operator()(ssize_t row, ssize_t col) const {
    return get_value(row, col);
}

template <typename ElemType>
ElemType * Buffer<ElemType>::data_ptr() const {
    return m_buffer;
}
template <typename ElemType>
ElemType const * Buffer<ElemType>::cdata_ptr() const {
    return const_cast<ElemType const *>(m_buffer);
}

template <typename ElemType>
Device Buffer<ElemType>::device() const {
    return m_device;
}
template <typename ElemType>
shape_t Buffer<ElemType>::shape() const {
    return {m_nrow, m_ncol};
}
template <typename ElemType>
ssize_t Buffer<ElemType>::nrow() const {
    return m_nrow;
}
template <typename ElemType>
ssize_t Buffer<ElemType>::ncol() const {
    return m_ncol;
}
template <typename ElemType>
ssize_t Buffer<ElemType>::num_elem() const {
    return m_nrow * m_ncol;
}
template <typename ElemType>
ssize_t Buffer<ElemType>::buffer_size() const {
    return m_nrow * m_ncol * sizeof(ElemType);
}
template <typename ElemType>
ssize_t Buffer<ElemType>::index_at(ssize_t row, ssize_t col) const {
    return row * m_ncol + col;
}
template <typename ElemType>
bool Buffer<ElemType>::is_same_shape(Buffer<ElemType> const &other) const {
    return m_nrow == other.nrow() && m_ncol == other.ncol();
}
template <typename ElemType>
bool Buffer<ElemType>::is_same_device(Buffer<ElemType> const &other) const {
    return m_device == other.device();
}
template <typename ElemType>
std::string Buffer<ElemType>::to_string() const {
    std::stringstream ss;
    ss << "<Buffer shape=(" << m_nrow << ", " << m_ncol << ") device=" << ((m_device == Device::CPU)? "CPU": "GPU") << " @" << (uintptr_t)this << ">";
    return ss.str();
}

template <typename ElemType>
void Buffer<ElemType>::to(Device device){
    if (m_device == device)
        return;
    ElemType *new_buffer;
    switch (m_device)
    {
    case Device::CPU:
        cudaMalloc(&new_buffer, buffer_size());
        cudaMemcpy(new_buffer, m_buffer, buffer_size(), cudaMemcpyHostToDevice);
        delete[] m_buffer;
        break;
    case Device::GPU:
        new_buffer = new ElemType[num_elem()];
        cudaMemcpy(new_buffer, m_buffer, buffer_size(), cudaMemcpyDeviceToHost);
        cudaFree(m_buffer);
        break;
    }
    m_buffer = new_buffer;
    m_device = device;
}
template <typename ElemType>
void Buffer<ElemType>::fill(ElemType val){
    switch (m_device)
    {
    case Device::CPU:
        std::fill_n(m_buffer, num_elem(), val);
        break;
    case Device::GPU:
        thrust::device_ptr<ElemType> dev_ptr(m_buffer);
        thrust::fill_n(dev_ptr, num_elem(), val); 
        break;
    }
}
template <typename ElemType>
void Buffer<ElemType>::clear(){
    _release();
    m_buffer = nullptr;
    m_nrow   = 0;
    m_ncol   = 0;
}

template <typename ElemType>
void Buffer<ElemType>::_release(){
    if (!m_buffer)
        return;
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

template <typename ElemType>
void Buffer<ElemType>::copy_to_numpy(ndarray_t<ElemType> &out) const {
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
        break;
    }
}
template <typename ElemType>
void Buffer<ElemType>::copy_from_numpy(ndarray_t<ElemType> const &src) const {
    ssize_t buf_buffer_size = buffer_size();
    ssize_t npy_buffer_size = src.nbytes();

    if (npy_buffer_size != buf_buffer_size)
        throw std::runtime_error("Size of numpy array does not match this buffer.");
    
    switch (m_device)
    {
    case Device::CPU:
        memcpy(m_buffer, src.data(), buf_buffer_size);
        break;
    case Device::GPU:
        cudaMemcpy(m_buffer, src.data(), buf_buffer_size, cudaMemcpyHostToDevice);
        break;
    }
}
// end Buffer

// start CURANDStates
CURANDStates::CURANDStates(ssize_t size, unsigned long long seed) 
    : m_size(size) 
{
    curand_setup(size, seed, &m_buffer);
}
CURANDStates::CURANDStates(CURANDStates const &other)
    : m_size(other.m_size)
{
    cudaMalloc(&m_buffer, buffer_size());
    cudaMemcpy(m_buffer, other.m_buffer, buffer_size(), cudaMemcpyDeviceToDevice);
}
CURANDStates::CURANDStates(CURANDStates &&other) noexcept
    : m_size(other.m_size), m_buffer(other.m_buffer)
{
    other.m_buffer = nullptr;
    other.m_size   = 0;
}
CURANDStates & CURANDStates::operator=(CURANDStates const &other){
    if (m_size != other.m_size)
        throw std::runtime_error("Shapes do not match.");
    cudaMemcpy(m_buffer, other.m_buffer, buffer_size(), cudaMemcpyDeviceToDevice);
    m_size = other.m_size;
    return *this;
}
CURANDStates & CURANDStates::operator=(CURANDStates &&other) noexcept {
    _release();
    m_buffer       = other.m_buffer;
    m_size         = other.m_size;
    other.m_buffer = nullptr;
    other.m_size   = 0;
    return *this;
}
CURANDStates::~CURANDStates(){
    _release();
}

cuda_rng_t * CURANDStates::data_ptr() const {
    return m_buffer;
}
cuda_rng_t const * CURANDStates::cdata_ptr() const {
    return const_cast<cuda_rng_t const *>(m_buffer);
}

ssize_t CURANDStates::num_elem() const {
    return m_size;
}
ssize_t CURANDStates::buffer_size() const {
    return m_size * sizeof(cuda_rng_t);
}
std::string CURANDStates::to_string() const {
    std::stringstream ss;
    ss << "<CURANDStates size=" << m_size << " device=GPU @" << (uintptr_t)this << ">";
    return ss.str();
}

void CURANDStates::clear(){
    _release();
    m_buffer = nullptr;
    m_size   = 0;
}

void CURANDStates::_release(){
    if (!m_buffer)
        return;
    curand_destroy(m_buffer);
}
// end CURANDStates