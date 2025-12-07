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
    // --- Row handled per block ---
    int row = blockIdx.y;  
    if (row >= M) return;

    // --- Column handled per thread ---
    int col = blockIdx.x * TILE_COLS + threadIdx.x;
    if (col >= K) return;

    extern __shared__ float sB[];

    int row_start = A_row_ptr[row];
    int row_end   = A_row_ptr[row + 1];

    float sum = 0.f;

    // Loop over N dimension (columns of A = rows of B)
    for (int j = row_start; j < row_end; ++j)
    {
        int a_col = A_col_idx[j];
        float a_val = A_values[j];

        // --- Load tile of B[a_col, :] into shared memory ---
        sB[threadIdx.x] = (col < K ? __ldg(&B[a_col * K + col]) : 0.0f);
        __syncthreads();

        // Every thread multiplies its column
        sum += a_val * sB[threadIdx.x];
        __syncthreads();
    }

    // Write output
    float old = (beta != 0.f ? C[row * K + col] : 0.f);
    C[row * K + col] = alpha * sum + beta * old;
}

} // namespace csrspmm::kernel
