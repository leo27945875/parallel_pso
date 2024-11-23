cmake_minimum_required(VERSION 3.31)

project(cuPSO LANGUAGES CUDA CXX)

find_package(CUDAToolkit 12.1 REQUIRED)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

include(CheckLanguage)

# Set include directories:
include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/csrc/buffer/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/csrc/utils/include)

# CUDA settings:
check_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 61)
message("CMAKE_CUDA_COMPILER : ${CMAKE_CUDA_COMPILER}")
message("CMAKE_CUDA_COMPILER_VERSION : ${CMAKE_CUDA_COMPILER_VERSION}")
message("CMAKE_CUDA_COMPILER_TOOLKIT_ROOT : ${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}")
message("CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message("CUDAToolkit_INCLUDE_DIRS : ${CUDAToolkit_INCLUDE_DIRS}")

# Create Python package:
set(package_install_dir ${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_PROJECT_NAME})
file(MAKE_DIRECTORY ${package_install_dir})
file(WRITE ${package_install_dir}/__init__.py "from .cuPSO import *")

# Compile settings:
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_FLAGS  "-O3 -Wall")
set(CMAKE_CUDA_FLAGS "-O3 -Wall")
set(LIBRARY_OUTPUT_PATH ${package_install_dir})

# Find source code:
file(GLOB_RECURSE cuda_utils_src_list  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/utils/*.cu )
file(GLOB_RECURSE cuda_buffer_src_list ${CMAKE_CURRENT_SOURCE_DIR}/csrc/buffer/*.cu)
file(GLOB_RECURSE cxx_src_list         ${CMAKE_CURRENT_SOURCE_DIR}/csrc/bind.cpp   )
set(src_list ${cuda_buffer_src_list} ${cuda_utils_src_list} ${cxx_src_list})
message("SRC_FILES:\n  ${src_list}")

# Build pybind11 shared library:
pybind11_add_module(${CMAKE_PROJECT_NAME} ${src_list})
