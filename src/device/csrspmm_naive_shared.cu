#include "kernel_utils.cuh"

namespace csrspmm::kernel {

__global__
void csrspmm_naive_shared(
    int M, int N, int K,
    float alpha,
    float beta,
    const float* __restrict__ A_values,
    const int* __restrict__ A_col_idx,
    const int* __restrict__ A_row_ptr,
    const float* __restrict__ B,
    float* __restrict__ C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= K) return;

    __shared__ float Btile[32 * 32];

    int row_start = A_row_ptr[row];
    int row_end   = A_row_ptr[row + 1];

    float sum = 0.0f;

    int s_idx = threadIdx.y * blockDim.x + threadIdx.x;

    for (int j = row_start; j < row_end; ++j) {
        int   A_col = A_col_idx[j];
        float A_val = A_values[j];

        float b_val = __ldg(&B[A_col * K + col]);

        Btile[s_idx] = b_val;
        __syncthreads();

        sum += A_val * Btile[s_idx];
        __syncthreads();
    }

    C[row * K + col] = alpha * sum + (beta != 0.0f ? beta * C[row * K + col] : 0.0f);
}

} // namespace csrspmm::kernel
