#include <cuda_runtime.h>
#include "csr_spmm.cuh"
#include "csr_utils.cuh"


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


__global__ void csr_spmm_warp_per_row(
    int M, int N, int K,
    float alpha, float beta,
    const float* __restrict__ A_values,
    const int*  __restrict__ A_col_idx,
    const int*  __restrict__ A_row_ptr,
    const float* __restrict__ B,
    float* __restrict__ C)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane    = threadIdx.x & (WARP_SIZE - 1);   // threadIdx.x % WARP_SIZE

    if (warp_id >= M) return;

    int row_start = A_row_ptr[warp_id];
    int row_end   = A_row_ptr[warp_id + 1];

    // iterate over columns of B
    for (int n = 0; n < N; ++n) {
        ACCUMULATOR_TYPE sum = 0.0f;

        // each thread processes part of row of A
        for (int j = row_start + lane; j < row_end; j += WARP_SIZE) {
            int col = A_col_idx[j];
            float val = A_values[j];
            sum += val * __ldg(&B[col * N + n]);
        }

        // sum all partials in warp using tree reduction 
        // NOTE: offset >>= 1 = offset / 2
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) 
            sum += __shfl_down_sync(0xffffffff, sum, offset);

        if (lane == 0)
            C[warp_id * N + n] = alpha * sum + beta * C[warp_id * N + n];
    }
}



// TODO:
// - problem with colescing access
// - buffering to share mem, block tiling on B
// - load balancing
