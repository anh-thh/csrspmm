#pragma once
#include <csrspmm/matrix.h>

namespace csrspmm::kernel {

__global__ 
void csrspmm_naive(
    int M, int N, int K,
    float alpha,
    float beta,
    const float* __restrict__ A_values,
    const int* __restrict__ A_col_idx,
    const int* __restrict__ A_row_ptr,
    const float* __restrict__ B,
    float* __restrict__ C);

__global__
void csrspmm_warp_per_row(
    int M, int N, int K,
    float alpha, float beta,
    const float* __restrict__ A_values,
    const int*  __restrict__ A_col_idx,
    const int*  __restrict__ A_row_ptr,
    const float* __restrict__ B,
    float* __restrict__ C);


__global__
void csrspmm_warp_per_row_fp4(
    int M, int N, int K,
    float alpha, float beta,
    const float* __restrict__ A_values,
    const int*  __restrict__ A_col_idx,
    const int*  __restrict__ A_row_ptr,
    const float* __restrict__ __align__(16) B, // Assume B_col is 16-byte (float4) aligned
    float* __restrict__ __align__(16) C);


__global__
void csrspmm_warp_per_row_smem(
    int M, int N, int K,
    float alpha, float beta,
    const float* __restrict__ A_values,
    const int*  __restrict__ A_col_idx,
    const int*  __restrict__ A_row_ptr,
    const int A_max_row_nnz,
    const float* __restrict__ B,   
    float* __restrict__ C);


__global__
void csrspmm_warp_per_row_smem_fp4(
    int M, int N, int K,
    float alpha, float beta,
    const float* __restrict__ A_values,
    const int*  __restrict__ A_col_idx,
    const int*  __restrict__ A_row_ptr,
    const int A_max_row_nnz,
    const float* __restrict__ __align__(16) B,   
    float* __restrict__ __align__(16) C);



__global__
void csrspmm_naive_shared(
    int M, int N, int K,
    float alpha, float beta,
    const float* __restrict__ A_values,
    const int* __restrict__ A_col_idx,
    const int* __restrict__ A_row_ptr,
    const float* __restrict__ B,
    float* __restrict__ C);

} // namespace csrspmm::kernel
