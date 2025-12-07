#include "kernel_utils.cuh"

namespace csrspmm::kernel {

#define TILE_COLS 128

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
    int col = blockIdx.x * TILE_COLS + threadIdx.x;

    if (row >= M || col >= K) return;

    extern __shared__ float sB[];

    int row_start = A_row_ptr[row];
    int row_end   = A_row_ptr[row + 1];

    float sum = 0.f;

    for (int j = row_start; j < row_end; j++) {
        int a_col = A_col_idx[j];
        float a_val = A_values[j];

        if (col < K)
            sB[threadIdx.x] = __ldg(&B[a_col * K + col]);
        __syncthreads();

        sum += a_val * sB[threadIdx.x];
        __syncthreads();
    }

    float old = beta != 0.f ? C[row * K + col] : 0.f;
    C[row * K + col] = alpha * sum + beta * old;
}

} // namespace csrspmm::kernel
