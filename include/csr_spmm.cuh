#pragma once
#include "csr_utils.cuh"
#include <cuda_runtime.h>


__global__ void csr_spmm_naive(
    int M, int N, int K,
    float alpha, 
    float beta, 
    const float* __restrict__ A_values, 
    const int* __restrict__ A_col_idx, 
    const int* __restrict__ A_row_ptr, 
    const float* __restrict__ B, 
    float* __restrict__ C);


