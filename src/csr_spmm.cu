#include <cuda_runtime.h>
#include "csr_spmm.cuh"
#include "csr_utils.cuh"

#define ACCUMULATOR_TYPE float


__global__ void csr_spmm_naive(
    int M, int N, int K,
    float alpha, 
    float beta, 
    const float* __restrict__ A_values, 
    const int* __restrict__ A_col_idx, 
    const int* __restrict__ A_row_ptr, 
    const float* __restrict__ B, 
    float* __restrict__ C) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    // each thread handles one row of A
    #pragma unroll
    for (int n = 0; n < N; ++n) {
        ACCUMULATOR_TYPE sum = 0.0f;

        // process non-zero elements in row
        int row_start = A_row_ptr[row];
        int row_end   = A_row_ptr[row + 1];

        for (int j = row_start; j < row_end; ++j) {
            int col = A_col_idx[j];
            float val = A_values[j];
            // sum += val * B[col * N + n];
            sum += val * __ldg(&B[col * N + n]);
        }

        C[row * N + n] = static_cast<float>(alpha * sum + beta * C[row * N + n]);
    }
}


