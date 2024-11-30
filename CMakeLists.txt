cmake_minimum_required(VERSION 3.31)

project(cuPSO LANGUAGES CUDA CXX)

# Helper functions:
function(print_target_include_directories T)
    get_property(dirs TARGET ${T} PROPERTY INCLUDE_DIRECTORIES)
    foreach(dir ${dirs})
        message("  '${dir}'")
    endforeach()
endfunction()

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
set(CMAKE_CXX_FLAGS "-O3 -Wall")

# Create Python package:
set(package_install_dir ${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_PROJECT_NAME})
set(LIBRARY_OUTPUT_PATH ${package_install_dir})
file(MAKE_DIRECTORY ${package_install_dir})
file(WRITE ${package_install_dir}/__init__.py "from .cuPSO import *")

# Compile settings:
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_BUILD_TYPE "Release")

# Find source code:
file(GLOB_RECURSE cuda_src_list ${CMAKE_CURRENT_SOURCE_DIR}/csrc/${CMAKE_PROJECT_NAME}/**/*.cu)
file(GLOB_RECURSE bind_src_list ${CMAKE_CURRENT_SOURCE_DIR}/csrc/${CMAKE_PROJECT_NAME}/bind.cpp)
message("CUDA Source files:\n  ${cuda_src_list}")
message("C++ Source files:\n  ${bind_src_list}")

# Build CUDA static library:
add_library(core STATIC ${cuda_src_list})
set_target_properties(core 
                      PROPERTIES
                      OUTPUT_NAME "${CMAKE_PROJECT_NAME}"
                      POSITION_INDEPENDENT_CODE ON
                      CUDA_SEPARABLE_COMPILATION ON 
                      CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(core PRIVATE ${pybind11_INCLUDE_DIRS})

# Build pybind11 shared library:
pybind11_add_module(${CMAKE_PROJECT_NAME} ${bind_src_list})
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE core)

# Print all include directories:
message("CUDA include directories:")
print_target_include_directories(core)
message("Project include directories:")
print_target_include_directories(${CMAKE_PROJECT_NAME})