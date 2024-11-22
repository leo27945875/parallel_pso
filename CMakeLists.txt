cmake_minimum_required(VERSION 3.31)

project(cuPSO LANGUAGES CUDA CXX)

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

include(CheckLanguage)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/csrc/buffer/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/csrc/utils/include)

set(package_install_dir ${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_PROJECT_NAME})
file(MAKE_DIRECTORY ${package_install_dir})
file(WRITE ${package_install_dir}/__init__.py "from .cuPSO import *")

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -Wall")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -Wall")
set(LIBRARY_OUTPUT_PATH ${package_install_dir})

file(GLOB_RECURSE cuda_src_list ${CMAKE_CURRENT_SOURCE_DIR}/csrc/buffer/*.cu)
file(GLOB_RECURSE cxx_src_list  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/bind.cpp)
set(src_list ${cuda_src_list} ${cxx_src_list})

check_language(CUDA)
message("CMAKE_CUDA_COMPILER_ID : ${CMAKE_CUDA_COMPILER_ID}")
message("CMAKE_CUDA_COMPILER_VERSION : ${CMAKE_CUDA_COMPILER_VERSION}")
message("SRC_FILES:\n    ${src_list}")

pybind11_add_module(${CMAKE_PROJECT_NAME} ${src_list})