#include "kernel_utils.cuh"

namespace csrspmm::kernel {


/**
 * Each thread load full row of A and full column of B => produce one output entry of C
 * finer-grained parallelism across both dimensions
 * require 2D block and grid to run
 */
__global__ 
void csr_spmm_naive(
    int M, int N, int K,
    float alpha,
    float beta,
    const float* __restrict__ A_values,
    const int* __restrict__ A_col_idx,
    const int* __restrict__ A_row_ptr,
    const float* __restrict__ B,
    float* __restrict__ C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // row in C
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // column in C

    if (row >= M || col >= K) return;

    int row_start = A_row_ptr[row];
    int row_end   = A_row_ptr[row + 1];

    float sum = 0.0f;
    for (int j = row_start; j < row_end; ++j) {
        int A_col   = A_col_idx[j];
        float A_val = A_values[j];
        sum        += A_val * __ldg(&B[A_col * K + col]);
    }
    
    // C[row * K + col] = alpha * sum + beta * C[row * K + col];
    C[row * K + col] = alpha * sum + (beta != 0.0f ? beta * C[row * K + col] : 0.0f);
}


} // namespace csrspmm::kernel

