#pragma once
#include "csr_utils.cuh"
#include <cuda_runtime.h>

#define ACCUMULATOR_TYPE float

__global__ void csr_spmm_naive(
    int M, int N, int K,
    float alpha, 
    float beta, 
    const float* __restrict__ A_values, 
    const int* __restrict__ A_col_idx, 
    const int* __restrict__ A_row_ptr, 
    const float* __restrict__ B, 
    float* __restrict__ C);


#define WARP_SIZE 32
__global__ void csr_spmm_warp_per_row(
    int M, int N, int K,
    float alpha, float beta,
    const float* __restrict__ A_values,
    const int*  __restrict__ A_col_idx,
    const int*  __restrict__ A_row_ptr,
    const float* __restrict__ B,
    float* __restrict__ C);
