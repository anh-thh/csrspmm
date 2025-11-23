#include "kernel_utils.cuh"
#include "csrspmm/config.h"

namespace csrspmm::kernel {

/**
 * Since all threads in a warp reuse the same A row, we stage its values and
 * column indices in per-warp shared memory to reduce global memory traffic.
 * All remaining logic follows the warp_per_row implementation.
 */
__global__
void csrspmm_warp_per_row_smem(
    int M, int N, int K,
    float alpha, float beta,
    const float* __restrict__ A_values,
    const int*  __restrict__ A_col_idx,
    const int*  __restrict__ A_row_ptr,
    const int A_max_row_nnz,
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
    int row_nnz   = row_end - row_start;

    // sharemem
    extern __shared__ unsigned char smem[];
    size_t per_warp_stride = A_max_row_nnz * (sizeof(float) + sizeof(int));
    unsigned char* warp_smem = smem + warp_in_block * per_warp_stride;

    float* s_vals = reinterpret_cast<float*>(warp_smem);
    int*   s_cols = reinterpret_cast<int*>(warp_smem + A_max_row_nnz * sizeof(float));

    for (int i = lane; i < row_nnz; i += WARP_SIZE) {
        s_vals[i] = A_values[row_start + i];
        s_cols[i] = A_col_idx[row_start + i];
    }
    __syncwarp();

    for (int col = lane; col < K; col += WARP_SIZE) {
        float sum = 0.0f;

        for (int j = 0; j < row_nnz; j++) {
            float a = s_vals[j];
            int   c = s_cols[j];

            const float* B_row = B + c * K;
            sum += a * __ldg(&B_row[col]);
        }

        float old = (beta != 0.0f ? C[row * K + col] : 0.0f);
        C[row * K + col] = alpha * sum + beta * old;
    }
}



}
