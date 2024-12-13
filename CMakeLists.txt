cmake_minimum_required(VERSION 3.31)

project(parallel_pso LANGUAGES CUDA CXX)

# Helper functions:
function(print_target_include_directories T)
    get_property(dirs TARGET ${T} PROPERTY INCLUDE_DIRECTORIES)
    foreach(dir ${dirs})
        message("  '${dir}'")
    endforeach()
endfunction()

# Package names:
set(cuPSO_package_name "cuPSO")
set(ompPSO_package_name "ompPSO")
set(pthreadPSO_package_name "pthreadPSO")

# Load packages:
find_package(CUDAToolkit 12.1 REQUIRED)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# Set include directories:
file(GLOB_RECURSE proj_inc_dirs LIST_DIRECTORIES true ${CMAKE_CURRENT_SOURCE_DIR}/csrc/**/include)
set(inc_dirs ${proj_inc_dirs} ${CUDAToolkit_INCLUDE_DIRS})
include_directories(${inc_dirs})

# CUDA settings:
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 61)
set(CMAKE_CUDA_FLAGS "-O3 -Wall")
message("CMAKE_CUDA_COMPILER : ${CMAKE_CUDA_COMPILER}")
message("CMAKE_CUDA_COMPILER_VERSION : ${CMAKE_CUDA_COMPILER_VERSION}")
message("CMAKE_CUDA_COMPILER_TOOLKIT_ROOT : ${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}")
message("CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message("CUDAToolkit_INCLUDE_DIRS : ${CUDAToolkit_INCLUDE_DIRS}")

# C++ settings:
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 -Wall -fopenmp -pthread")

# Other compile settings:
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_BUILD_TYPE "Release")

# Build cuPSO:
# ----- Create Python package -----
set(cuPSO_package_install_dir ${CMAKE_CURRENT_SOURCE_DIR}/${cuPSO_package_name})
file(MAKE_DIRECTORY ${cuPSO_package_install_dir})
file(WRITE ${cuPSO_package_install_dir}/__init__.py "from .${cuPSO_package_name} import *")
# ----- Find source code -----
file(GLOB_RECURSE cuPSO_cuda_src_list ${CMAKE_CURRENT_SOURCE_DIR}/csrc/${cuPSO_package_name}/**/*.cu)
file(GLOB_RECURSE cuPSO_bind_src_list ${CMAKE_CURRENT_SOURCE_DIR}/csrc/${cuPSO_package_name}/bind.cpp)
# ----- Build CUDA static library -----
add_library(core STATIC ${cuPSO_cuda_src_list})
set_target_properties(core 
                      PROPERTIES
                      OUTPUT_NAME "${cuPSO_package_name}"
                      ARCHIVE_OUTPUT_DIRECTORY "${cuPSO_package_install_dir}"
                      POSITION_INDEPENDENT_CODE ON
                      CUDA_SEPARABLE_COMPILATION ON 
                      CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(core PRIVATE ${pybind11_INCLUDE_DIRS})
# ----- Build cuPSO pybind11 shared library -----
pybind11_add_module(${cuPSO_package_name} ${cuPSO_bind_src_list})
target_link_libraries(${cuPSO_package_name} PRIVATE core)
set_target_properties(${cuPSO_package_name} 
                      PROPERTIES 
                      LIBRARY_OUTPUT_DIRECTORY "${cuPSO_package_install_dir}")

# Build ompPSO:
# ----- Create Python package -----
set(ompPSO_package_install_dir ${CMAKE_CURRENT_SOURCE_DIR}/${ompPSO_package_name})
file(MAKE_DIRECTORY ${ompPSO_package_install_dir})
file(WRITE ${ompPSO_package_install_dir}/__init__.py "from .${ompPSO_package_name} import *")
# ----- Find source code -----
file(GLOB_RECURSE ompPSO_bind_src_list ${CMAKE_CURRENT_SOURCE_DIR}/csrc/${ompPSO_package_name}/ompPSO.cpp)
# ----- Build ompPSO pybind11 shared library -----
pybind11_add_module(${ompPSO_package_name} ${ompPSO_bind_src_list})
set_target_properties(${ompPSO_package_name}
                      PROPERTIES
                      LIBRARY_OUTPUT_DIRECTORY "${ompPSO_package_install_dir}")

# Build pthreadPSO:
# ----- Create Python package -----
set(pthreadPSO_package_install_dir ${CMAKE_CURRENT_SOURCE_DIR}/${pthreadPSO_package_name})
file(MAKE_DIRECTORY ${pthreadPSO_package_install_dir})
file(WRITE ${pthreadPSO_package_install_dir}/__init__.py "from .${pthreadPSO_package_name} import *")
# ----- Find source code -----
file(GLOB_RECURSE pthreadPSO_bind_src_list ${CMAKE_CURRENT_SOURCE_DIR}/csrc/${pthreadPSO_package_name}/pthreadPSO.cpp)
# ----- Build pthreadPSO pybind11 shared library -----
pybind11_add_module(${pthreadPSO_package_name} ${pthreadPSO_bind_src_list})
set_target_properties(${pthreadPSO_package_name}
                      PROPERTIES
                      LIBRARY_OUTPUT_DIRECTORY "${pthreadPSO_package_install_dir}")

# Print all include directories:
message("CUDA include directories:")
print_target_include_directories(core)