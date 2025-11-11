#include <cuda_runtime.h>
#include "csr_spmm.cuh"
#include "csr_utils.cuh"


__global__ void spmm_csr_dense_naive(
    int M, int N, int K,
    float alpha, CSRMatrix A, float* B, 
    float beta, float* C) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    // each thread handles one row of A
    for (int n = 0; n < N; ++n) {
        double sum = 0.0f;

        // process non-zero elements in row
        int row_start = A.row_ptr[row];
        int row_end   = A.row_ptr[row + 1];

        for (int j = row_start; j < row_end; ++j) {
            int col = A.col_idx[j];
            float val = A.values[j];
            sum += val * B[col * N + n];
        }

        C[row * N + n] = (float) alpha * sum + beta * C[row * N + n];
    }
}

