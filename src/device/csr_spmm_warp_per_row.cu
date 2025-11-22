#include "kernel_utils.cuh"
#include "csrspmm/config.h"

namespace csrspmm::kernel {


/**
 * Each warp loads one sparse row of A and processes WARP_SIZE (32) columns of B.
 * Lane k (of the warp) computes output columns {k, k+32, k+64, ...}
 * => producing C[row, lane::32].
 * Only need 1D grid and block to run (energy consumption should be considered ??)
 */
__global__
void csr_spmm_warp_per_row(
    int M, int N, int K,
    float alpha, float beta,
    const float* __restrict__ A_values,
    const int*  __restrict__ A_col_idx,
    const int*  __restrict__ A_row_ptr,
    const float* __restrict__ B,
    float* __restrict__ C)
{
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int warp_in_block   = threadIdx.x / WARP_SIZE;
    const int lane            = lane_id();

    int row = blockIdx.x * warps_per_block + warp_in_block;
    if (row >= M) return;

    int row_start = A_row_ptr[row];
    int row_end   = A_row_ptr[row + 1];

    // Each lane computes for columns: lane, lane+32, lane+64, ...
    for (int col = lane; col < K; col += WARP_SIZE) {
        float sum = 0.0f;

        for (int j = row_start; j < row_end; j++) {
            float a = __ldg(&A_values[j]);
            int   c = __ldg(&A_col_idx[j]);

            const float* B_row = B + c * K;
            sum += a * __ldg(&B_row[col]);
        }

        float old = (beta != 0.0f) ? C[row * K + col] : 0.0f;
        C[row * K + col] = alpha * sum + beta * old;
    }
}


}
