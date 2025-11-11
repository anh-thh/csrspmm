#pragma once
#include "csr_utils.cuh"
#include <cuda_runtime.h>


__global__ void spmm_csr_dense_naive(
    int M, int N, int K,
    float alpha, CSRMatrix A, float* B, 
    float beta, float* C);


