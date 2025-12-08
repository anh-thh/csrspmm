cmake_minimum_required(VERSION 3.20)
project(csrspmm LANGUAGES C CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

# ------------ CUDA SETTINGS -------------------
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Enable relocatable device code (needed for separate compilation)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Change this to your CUDA architecture (for better compiler optmizatiom)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

# ------------- Optimization Flags -----------
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra")
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -G -g")


# ------------ INCLUDE ---------------------
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src/device)
include_directories(${PROJECT_SOURCE_DIR}/src/host)

# ------------ SOURCES --------------------
file(GLOB HOST_CPP_SRCS
    src/host/*.cpp
)

file(GLOB HOST_CU_SRCS
    src/host/*.cu
)

file(GLOB DEVICE_CU_SRCS
    src/device/*.cu
)

set(CSRSPMM_SRCS
    src/csrspmm.cpp
    ${HOST_CPP_SRCS}
    ${HOST_CU_SRCS}
    ${DEVICE_CU_SRCS}
)

# ------------ BUILD LIBRARY -------------------
add_library(csrspmm STATIC ${CSRSPMM_SRCS})

target_include_directories(csrspmm PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(csrspmm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ------------ Testbenches -----------------
add_executable(test_csrspmm tests/test_csrspmm.cu)

target_link_libraries(test_csrspmm PRIVATE csrspmm)

set_target_properties(test_csrspmm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)   
# required if there are multiple CUDA files, allow device code in 
# separate .cu files to call each other

# ------------ Benchmark CSR kernels -----------------
add_executable(bench_csrspmm benchmarks/bench_csrspmm.cu)

target_include_directories(bench_csrspmm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)

target_link_libraries(bench_csrspmm PRIVATE csrspmm)

set_target_properties(bench_csrspmm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)   


# ------------ Benchmark curparse API -----------------
add_executable(bench_cusparse benchmarks/bench_cusparse.cu)

target_include_directories(bench_cusparse PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)

target_link_libraries(bench_cusparse PRIVATE 
    csrspmm
    -lcusparse
)

set_target_properties(bench_cusparse PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)


configure_file(
    ${CMAKE_SOURCE_DIR}/benchmarks/exp1.sh
    ${CMAKE_BINARY_DIR}/exp1.sh
    COPYONLY
)

