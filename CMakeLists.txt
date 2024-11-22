cmake_minimum_required(VERSION 3.31)

project(try_matrix LANGUAGES CUDA CXX)

include(CheckLanguage)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -Wall")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -Wall")
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

file(GLOB_RECURSE cuda_src_list ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cu)
file(GLOB_RECURSE cxx_src_list ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp)
set(src_list ${cuda_src_list} ${cxx_src_list})

check_language(CUDA)
message("CMAKE_CUDA_COMPILER_ID : ${CMAKE_CUDA_COMPILER_ID}")
message("CMAKE_CUDA_COMPILER_VERSION : ${CMAKE_CUDA_COMPILER_VERSION}")
message("SRC_FILES: ${src_list}")

# pybind11_add_module(_matrix ${src_list})