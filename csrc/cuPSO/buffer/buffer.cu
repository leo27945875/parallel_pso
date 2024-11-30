#include <iomanip>
#include <sstream>
#include <algorithm>
#include <stdexcept>

#include <thrust/fill.h>
#include <thrust/device_ptr.h>

#include "buffer.cuh"


__global__ void cuda_fill_2d_kernel(scalar_t *data_ptr, ssize_t nrow, ssize_t ncol, ssize_t pitch, scalar_t val){
    ssize_t xidx = blockDim.x * blockIdx.x + threadIdx.x;
    ssize_t yidx = blockDim.y * blockIdx.y + threadIdx.y;
    for (ssize_t r = xidx; r < nrow; r += gridDim.x * blockDim.x)
    for (ssize_t c = yidx; c < ncol; c += gridDim.y * blockDim.y){
        *(scalar_t *)((char *)data_ptr + r * pitch + c * sizeof(scalar_t)) = val;
    }
}

static void cpu_malloc_func(scalar_t **ptr, ssize_t nrow, ssize_t ncol, ssize_t *pitch){
    *ptr = new scalar_t[nrow * ncol];
    *pitch = ncol * sizeof(scalar_t);
}

static void cpu_memcpy_func(scalar_t *dst, scalar_t const *src, ssize_t nrow, ssize_t ncol){
    memcpy(dst, src, nrow * ncol * sizeof(scalar_t));
}

static void cuda_malloc_func(scalar_t **ptr, ssize_t nrow, ssize_t ncol, ssize_t *pitch){
#if IS_CUDA_ALIGN_MALLOC
    if (nrow == 1 || ncol == 1){
        cudaMalloc(ptr, ncol * nrow * sizeof(scalar_t));
        *pitch = ncol * sizeof(scalar_t);
    }else{
        cudaMallocPitch(ptr, reinterpret_cast<size_t*>(pitch), ncol * sizeof(scalar_t), nrow);
    }
#else
    cudaMalloc(ptr, ncol * nrow * sizeof(scalar_t));
    *pitch = ncol * sizeof(scalar_t);
#endif
    cudaCheckErrors("Failed to allocate GPU buffer.");
}
static void cuda_malloc_func(cuda_rng_t **ptr, ssize_t nrow, ssize_t ncol, ssize_t *pitch){
#if IS_CUDA_ALIGN_MALLOC
    if (nrow == 1 || ncol == 1){
        cudaMalloc(ptr, ncol * nrow * sizeof(cuda_rng_t));
        *pitch = ncol * sizeof(cuda_rng_t);
    }else{
        cudaMallocPitch(ptr, reinterpret_cast<size_t*>(pitch), ncol * sizeof(cuda_rng_t), nrow);
    }
#else
    cudaMalloc(ptr, ncol * nrow * sizeof(cuda_rng_t));
    *pitch = ncol * sizeof(cuda_rng_t);
#endif
    cudaCheckErrors("Failed to allocate GPU buffer.");
}

static void cuda_memcpy_func(scalar_t *dst, scalar_t const *src, ssize_t nrow, ssize_t ncol, ssize_t dst_pitch, ssize_t src_pitch, cudaMemcpyKind kind){
#if IS_CUDA_ALIGN_MALLOC
    cudaMemcpy2D(dst, dst_pitch, src, src_pitch, ncol * sizeof(scalar_t), nrow, kind);
#else
    cudaMemcpy(dst, src, ncol * nrow * sizeof(scalar_t), kind);
#endif
    cudaCheckErrors("Fail to copy memory.");
}
static void cuda_memcpy_func(cuda_rng_t *dst, cuda_rng_t const *src, ssize_t nrow, ssize_t ncol, ssize_t dst_pitch, ssize_t src_pitch, cudaMemcpyKind kind){
#if IS_CUDA_ALIGN_MALLOC
    cudaMemcpy2D(dst, dst_pitch, src, src_pitch, ncol * sizeof(cuda_rng_t), nrow, kind);
#else
    cudaMemcpy(dst, src, ncol * nrow * sizeof(cuda_rng_t), kind);
#endif
    cudaCheckErrors("Fail to copy memory.");
}

static void cuda_fill_2d(scalar_t *data_ptr, ssize_t nrow, ssize_t ncol, ssize_t pitch, scalar_t val){
#if IS_CUDA_ALIGN_MALLOC
    dim3 grid_dims(get_num_block_x(nrow), get_num_block_y(ncol));
    dim3 block_dims(BLOCK_DIM_X, BLOCK_DIM_Y);
    cuda_fill_2d_kernel<<<grid_dims, block_dims>>>(data_ptr, nrow, ncol, pitch, val);
#else
    thrust::fill_n(thrust::device_ptr<scalar_t>(data_ptr), nrow * ncol, val);
#endif
    cudaCheckErrors("Failed to fill buffer.");
}


// start Buffer
Buffer::Buffer(ssize_t nrow, ssize_t ncol, Device device)
    : m_nrow(nrow), m_ncol(ncol), m_device(device)
{
    if (nrow == 0 || ncol == 0){
        m_buffer = nullptr;
        return;
    }
    switch (m_device)
    {
    case Device::CPU:
        cpu_malloc_func(&m_buffer, nrow, ncol, &m_pitch);
        break;
    case Device::GPU:
        cuda_malloc_func(&m_buffer, nrow, ncol, &m_pitch);
        break;
    }
}
Buffer::Buffer(Buffer const &other)
    : m_nrow(other.m_nrow), m_ncol(other.m_ncol), m_device(other.m_device)
{
    switch (m_device)
    {
    case Device::CPU:
        cpu_malloc_func(&m_buffer, m_nrow, m_ncol, &m_pitch);
        cpu_memcpy_func(m_buffer, other.m_buffer, m_nrow, m_ncol);
        break;
    case Device::GPU:
        cuda_malloc_func(&m_buffer, m_nrow, m_ncol, &m_pitch);
        cuda_memcpy_func(m_buffer, other.m_buffer, m_nrow, m_ncol, m_pitch, other.m_pitch, cudaMemcpyDeviceToDevice);
        break;
    }
}
Buffer::Buffer(Buffer &&other) noexcept 
    : m_nrow(other.m_nrow), m_ncol(other.m_ncol), m_device(other.m_device), m_buffer(other.m_buffer), m_pitch(other.m_pitch)
{
    other.m_buffer = nullptr;
    other.m_pitch  = 0;
    other.m_nrow   = 0;
    other.m_ncol   = 0;
}
Buffer & Buffer::operator=(Buffer const &other){
    if (!is_same_shape(other))
        throw std::runtime_error("Shapes do not match.");
    if (!is_same_device(other))
        throw std::runtime_error("Devices do not match.");
    switch (m_device)
    {
    case Device::CPU:
        cpu_memcpy_func(m_buffer, other.m_buffer, m_nrow, m_ncol);
        break;
    case Device::GPU:
        cuda_memcpy_func(m_buffer, other.m_buffer, m_nrow, m_ncol, m_pitch, other.m_pitch, cudaMemcpyDeviceToDevice);
        break;
    }
    return *this;
}
Buffer & Buffer::operator=(Buffer &&other) noexcept {
    _release();
    m_buffer       = other.m_buffer;
    m_pitch        = other.m_pitch;
    m_nrow         = other.m_nrow;
    m_ncol         = other.m_ncol;
    m_device       = other.m_device;
    other.m_buffer = nullptr;
    other.m_pitch  = 0;
    other.m_nrow   = 0;
    other.m_ncol   = 0;
    return *this;
}
Buffer::~Buffer(){
    _release();
}

void Buffer::set_value(ssize_t row, ssize_t col, scalar_t val){
    switch (m_device)
    {
    case Device::CPU:
        *_ptr_at(row, col) = val;
        break;
    case Device::GPU:
        cudaMemcpy(_ptr_at(row, col), &val, sizeof(scalar_t), cudaMemcpyHostToDevice);
        break;
    }
}
scalar_t Buffer::get_value(ssize_t row, ssize_t col) const {
    scalar_t res;
    switch (m_device)
    {
    case Device::CPU:
        res = *_ptr_at(row, col);
        break;
    case Device::GPU:
        cudaMemcpy(&res, _ptr_at(row, col), sizeof(scalar_t), cudaMemcpyDeviceToHost);
        break;
    }
    return res;
}
scalar_t Buffer::operator()(ssize_t row, ssize_t col) const {
    return get_value(row, col);
}

scalar_t * Buffer::data_ptr() const {
    return m_buffer;
}
scalar_t const * Buffer::cdata_ptr() const {
    return m_buffer;
}

Device Buffer::device() const {
    return m_device;
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
    return m_nrow * m_ncol * sizeof(scalar_t);
}
ssize_t Buffer::padded_size() const {
    return m_nrow * m_pitch;
}
ssize_t Buffer::pitch() const {
    return m_pitch;
}
ssize_t Buffer::default_pitch() const {
    return m_ncol * sizeof(scalar_t);
}
bool Buffer::is_same_shape(Buffer const &other) const {
    return m_nrow == other.nrow() && m_ncol == other.ncol();
}
bool Buffer::is_same_device(Buffer const &other) const {
    return m_device == other.device();
}
std::string Buffer::to_string() const {
    std::stringstream ss;
    ss << "<Buffer shape=(" << m_nrow << ", " << m_ncol << ") device=" << ((m_device == Device::CPU)? "CPU": "GPU") << " pitch=" << m_pitch << " @" << (uintptr_t)this << ">";
    return ss.str();
}
std::string Buffer::to_elem_string() const {
    scalar_t *tmp_buffer = m_buffer;
    if (m_device == Device::GPU){
        tmp_buffer = new scalar_t[num_elem()];
        cuda_memcpy_func(tmp_buffer, m_buffer, m_nrow, m_ncol, default_pitch(), m_pitch, cudaMemcpyDeviceToHost);
    }
    std::stringstream ss;
    for (ssize_t i = 0; i < m_nrow; i++)
    for (ssize_t j = 0; j < m_ncol; j++){
        if (j == m_ncol - 1)
            ss << std::fixed << std::setprecision(8) << tmp_buffer[i * m_ncol + j] << "\n";
        else
            ss << std::fixed << std::setprecision(8) << tmp_buffer[i * m_ncol + j] << ", ";
    }
    if (m_device == Device::GPU){
        delete[] tmp_buffer;
    }
    return ss.str();
}
void Buffer::to(Device device){
    if (m_device == device)
        return;
    ssize_t   new_pitch  = 0L;
    scalar_t *new_buffer = nullptr; 
    switch (m_device)
    {
    case Device::CPU:
        cuda_malloc_func(&new_buffer, m_nrow, m_ncol, &new_pitch);
        cuda_memcpy_func(new_buffer, m_buffer, m_nrow, m_ncol, new_pitch, m_pitch, cudaMemcpyHostToDevice);
        delete[] m_buffer;
        break;
    case Device::GPU:
        cpu_malloc_func(&new_buffer, m_nrow, m_ncol, &new_pitch);
        cuda_memcpy_func(new_buffer, m_buffer, m_nrow, m_ncol, new_pitch, m_pitch, cudaMemcpyDeviceToHost);
        cudaFree(m_buffer);
        break;
    }
    m_buffer = new_buffer;
    m_pitch  = new_pitch;
    m_device = device;
}
void Buffer::fill(scalar_t val){
    switch (m_device)
    {
    case Device::CPU:
        std::fill_n(m_buffer, num_elem(), val);
        break;
    case Device::GPU:
        cuda_fill_2d(m_buffer, m_nrow, m_ncol, m_pitch, val);
        break;
    }
}
void Buffer::show() const {
    for (ssize_t i = 0; i < m_nrow; i++)
    for (ssize_t j = 0; j < m_ncol; j++){
        if (j == m_ncol - 1)
            std::cout << std::fixed << std::setprecision(8) << get_value(i, j) << std::endl;
        else
            std::cout << std::fixed << std::setprecision(8) << get_value(i, j) << ", ";
    }
}
void Buffer::clear(){
    _release();
    m_buffer = nullptr;
    m_pitch  = 0;
    m_nrow   = 0;
    m_ncol   = 0;
}

void Buffer::copy_to_numpy(ndarray_t<scalar_t> &out) const {
    if (out.nbytes() != buffer_size())
        throw std::runtime_error("Size of numpy array does not match this buffer.");
    
    switch (m_device)
    {
    case Device::CPU:
        cpu_memcpy_func(out.mutable_data(), m_buffer, m_nrow, m_ncol);
        break;
    case Device::GPU:
        cuda_memcpy_func(out.mutable_data(), m_buffer, m_nrow, m_ncol, default_pitch(), m_pitch, cudaMemcpyDeviceToHost);
        break;
    }
}
void Buffer::copy_from_numpy(ndarray_t<scalar_t> const &src){
    if (src.nbytes() != buffer_size())
        throw std::runtime_error("Size of numpy array does not match this buffer.");
    
    switch (m_device)
    {
    case Device::CPU:
        cpu_memcpy_func(m_buffer, src.data(), m_nrow, m_ncol);
        break;
    case Device::GPU:
        cuda_memcpy_func(m_buffer, src.data(), m_nrow, m_ncol, m_pitch, default_pitch(), cudaMemcpyHostToDevice);
        break;
    }
}
void Buffer::copy_to_buffer(Buffer &out) const {
    out = *this;
}
void Buffer::copy_from_buffer(Buffer const &src){
    *this = src;
}

scalar_t * Buffer::_ptr_at(ssize_t row, ssize_t col) const {
    return (scalar_t *)((char *)m_buffer + row * m_pitch + col * sizeof(scalar_t));
}
void Buffer::_release(){
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
// end Buffer

// start CURANDStates
CURANDStates::CURANDStates(unsigned long long seed, ssize_t nrow, ssize_t ncol) 
    : m_nrow(nrow), m_ncol(ncol) 
{
    curand_setup(nrow, ncol, seed, &m_buffer, &m_pitch);
}
CURANDStates::CURANDStates(CURANDStates const &other)
    : m_nrow(other.m_nrow), m_ncol(other.m_ncol)
{
    cuda_malloc_func(&m_buffer, m_nrow, m_ncol, &m_pitch);
    cuda_memcpy_func(m_buffer, other.m_buffer, m_nrow, m_ncol, m_pitch, other.m_pitch, cudaMemcpyDeviceToDevice);
}
CURANDStates::CURANDStates(CURANDStates &&other) noexcept
    : m_nrow(other.m_nrow), m_ncol(other.m_ncol), m_buffer(other.m_buffer), m_pitch(other.m_pitch)
{
    other.m_buffer = nullptr;
    other.m_pitch  = 0;
    other.m_nrow   = 0;
    other.m_ncol   = 0;
}
CURANDStates & CURANDStates::operator=(CURANDStates const &other){
    if (m_nrow != other.m_nrow || m_ncol != other.m_ncol)
        throw std::runtime_error("Shapes do not match.");
    cuda_memcpy_func(m_buffer, other.m_buffer, m_nrow, m_ncol, m_pitch, other.m_pitch, cudaMemcpyDeviceToDevice);
    return *this;
}
CURANDStates & CURANDStates::operator=(CURANDStates &&other) noexcept {
    _release();
    m_buffer       = other.m_buffer;
    m_pitch        = other.m_pitch;
    m_nrow         = other.m_nrow;
    m_ncol         = other.m_ncol;
    other.m_buffer = nullptr;
    other.m_pitch  = 0;
    other.m_nrow   = 0;
    other.m_ncol   = 0;
    return *this;
}
CURANDStates::~CURANDStates(){
    _release();
}

cuda_rng_t * CURANDStates::data_ptr() const {
    return m_buffer;
}
cuda_rng_t const * CURANDStates::cdata_ptr() const {
    return m_buffer;
}

ssize_t CURANDStates::num_elem() const {
    return m_nrow * m_ncol;
}
ssize_t CURANDStates::buffer_size() const {
    return m_nrow * m_ncol * sizeof(cuda_rng_t);
}
ssize_t CURANDStates::pitch() const {
    return m_pitch;
}
std::string CURANDStates::to_string() const {
    std::stringstream ss;
    ss << "<CURANDStates size=(" << m_nrow << ", " << m_ncol << ") pitch=" << m_pitch << " device=GPU @" << (uintptr_t)this << ">";
    return ss.str();
}

void CURANDStates::clear(){
    _release();
    m_buffer = nullptr;
    m_pitch  = 0;
    m_nrow   = 0;
    m_ncol   = 0;
}

void CURANDStates::_release(){
    if (!m_buffer)
        return;
    curand_destroy(m_buffer);
}
// end CURANDStates