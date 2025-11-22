#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cusparse.h>

namespace csrspmm {


inline void cudaErrorCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)
                  << " at " << file << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define cudaCheck(err) (cudaErrorCheck(err, __FILE__, __LINE__))


inline void cublasErrorCheck(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error: " << status
                  << " at " << file << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define cublasCheck(err) (cublasErrorCheck(err, __FILE__, __LINE__))


static inline void cusparseErrorCheck(cusparseStatus_t status, const char* file, int line) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "cuSPARSE error %d at %s:%d\n", int(status), file, line);
        std::exit(EXIT_FAILURE);
    }
}

#define cusparseCheck(err) (cusparseErrorCheck(err, __FILE__, __LINE__))

}

