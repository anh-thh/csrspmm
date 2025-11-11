#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

#define cudaCheck(err) (cudaErrorCheck(err, __FILE__, __LINE__))
#define cublasCheck(err) (cublasErrorCheck(err, __FILE__, __LINE__))
#define ROUND_UP_TO_NEAREST(M, N) (((M) + (N)-1) / (N))


inline void cudaErrorCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)
                  << " at " << file << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline void cublasErrorCheck(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error: " << status
                  << " at " << file << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}
