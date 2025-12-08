#include "kernel_utils.cuh"

namespace csrspmm::kernel {

#ifndef NAIVE_SHARED_MAX_ROW_NNZ
#define NAIVE_SHARED_MAX_ROW_NNZ 128   // how many nonzeros per chunk we cache
#endif

/**
 * Naive + shared:
 *  - 1 block = 1 row of C
 *  - threads in x dimension cover columns of C
 *  - we cache segments of the CSR row (col + value) in shared memory
 *
 * Shapes:
 *  A: M x N (CSR)
 *  B: N x K  (row-major)
 *  C: M x K  (row-major)
 *
 * Kernel params:
 *   M = A.height
 *   N = A.width
 *   K = B.width   (number of output columns)
 */
__global__
void csrspmm_naive_shared(
    int M, int N, int K,
    float alpha,
    float beta,
    const float* __restrict__ A_values,
    const int*   __restrict__ A_col_idx,
    const int*   __restrict__ A_row_ptr,
    const float* __restrict__ B,
    float*       __restrict__ C)
{
    // 1 block handles exactly one row
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= K) return;

    int row_start = A_row_ptr[row];
    int row_end   = A_row_ptr[row + 1];

    // cache a chunk of this CSR row in shared
    __shared__ int   s_cols[NAIVE_SHARED_MAX_ROW_NNZ];
    __shared__ float s_vals[NAIVE_SHARED_MAX_ROW_NNZ];

    float sum = 0.0f;

    // walk the row in chunks
    for (int base = row_start; base < row_end; base += NAIVE_SHARED_MAX_ROW_NNZ) {
        int chunk = row_end - base;
        if (chunk > NAIVE_SHARED_MAX_ROW_NNZ)
            chunk = NAIVE_SHARED_MAX_ROW_NNZ;

        // load (col, val) of this chunk into shared, cooperatively
        for (int i = threadIdx.x; i < chunk; i += blockDim.x) {
            s_cols[i] = A_col_idx[base + i];
            s_vals[i] = A_values[base + i];
        }

        __syncthreads();

        // compute contribution of this chunk
        #pragma unroll
        for (int i = 0; i < chunk; ++i) {
            int   A_col = s_cols[i];
            float A_val = s_vals[i];

            // B is N x K, row-major => stride is K
            float B_val = __ldg(&B[A_col * K + col]);
            sum += A_val * B_val;
        }

        __syncthreads();
    }

    int idx = row * K + col;
    float old = (beta != 0.0f ? C[idx] : 0.0f);
    C[idx] = alpha * sum + beta * old;
}

} // namespace csrspmm::kernel
