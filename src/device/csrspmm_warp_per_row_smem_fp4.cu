#include "kernel_utils.cuh"
#include "csrspmm/config.h"

namespace csrspmm::kernel {

/**
 * Since all threads in a warp reuse the same A row, we stage its values and
 * column indices in per-warp shared memory to reduce global memory traffic.
 * All remaining logic follows the warp_per_row implementation.
 */
__global__
void csrspmm_warp_per_row_smem_fp4(
    int M, int N, int K,
    float alpha, float beta,
    const float* __restrict__ A_values,
    const int*  __restrict__ A_col_idx,
    const int*  __restrict__ A_row_ptr,
    const int A_max_row_nnz,
    const float* __restrict__ __align__(16) B,   
    float* __restrict__ __align__(16) C)
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

    const int K_vec = K / 4;     // # of full float4 segments
    float4* Crow4 = reinterpret_cast<float4*>(C + row * K);

    for (int col4 = lane; col4 < K_vec; col4 += WARP_SIZE) {
        float4 acc = make_float4(0,0,0,0);

        for (int j = 0; j < row_nnz; j++) {
            float a = s_vals[j];
            int   c = s_cols[j];

            const float4* Brow4 = reinterpret_cast<const float4*>(B + c * K);
            float4 b = Brow4[col4];

            acc.x += a * b.x;
            acc.y += a * b.y;
            acc.z += a * b.z;
            acc.w += a * b.w;
        }

        if (beta != 0.f) {
            float4 o = Crow4[col4];
            Crow4[col4] = make_float4(
                alpha*acc.x + beta*o.x,
                alpha*acc.y + beta*o.y,
                alpha*acc.z + beta*o.z,
                alpha*acc.w + beta*o.w
            );
        } else {
            Crow4[col4] = make_float4(
                alpha*acc.x, alpha*acc.y,
                alpha*acc.z, alpha*acc.w
            );
        }
    }
}



}
