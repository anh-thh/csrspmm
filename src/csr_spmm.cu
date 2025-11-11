#include <cuda_runtime.h>
#include "csr_spmm.cuh"
#include "csr_utils.cuh"
#include "helper.cuh"


/*
 * Sparse x Dense Matrix Multiplication
 * A[M, N] x B[N, K] => C[M, K]
 */


void run_csr_spmm(Algo algo,
                  int M, int N, int K,
                  float alpha, float beta,
                  float* A_values,
                  int* A_col_idx,
                  int* A_row_ptr,
                  float *B, 
                  float *C) {
    switch (algo) {
    case cuSPARSELt: {
        // Placeholder for cuSPARSELt
        break;
    }

    case naive: {
        const int block_size = 32;
        const dim3 gridDim(ROUND_UP_TO_NEAREST(M, block_size));
        const dim3 blockDim(block_size);

        csr_spmm_naive<<<gridDim, blockDim>>>(M, N, K, 
                                              alpha, beta,
                                              A_values, 
                                              A_col_idx,  
                                              A_row_ptr,
                                              B, C);
        break;
    }

    case warp_per_row: {
        const int threads_per_block = 128;
        int warps_per_block   = threads_per_block / WARP_SIZE;
        int num_blocks        = ROUND_UP_TO_NEAREST(M, warps_per_block);
        dim3 blockDim(threads_per_block);
        dim3 gridDim(num_blocks);

        csr_spmm_warp_per_row<<<gridDim, blockDim>>>(M, N, K, alpha, beta,
                                                     A_values,
                                                     A_col_idx,
                                                     A_row_ptr,
                                                     B, C);
        break;
    }

    default:
        printf("Invalid algorithm: %d\n", algo);
        exit(EXIT_FAILURE);
    }

    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError());
}


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

    // this loop handle full row of C
    #pragma unroll 
    for (int n = 0; n < K; ++n) {
        float sum = 0.0f;               // NOTE: consider higher precison for accumulator 

        // non-zero elements in A's row
        int row_start = A_row_ptr[row];
        int row_end   = A_row_ptr[row + 1];

        for (int j = row_start; j < row_end; ++j) {
            int col = A_col_idx[j];
            float val = A_values[j];
            // sum += val * B[col * K + n];
            sum += val * __ldg(&B[col * K + n]);
        }

        C[row * K + n] = static_cast<float>(alpha * sum + beta * C[row * K + n]);
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
    for (int n = 0; n < K; ++n) {
        float sum = 0.0f;               // NOTE: consider higher precison for accumulator 

        // each thread processes part of row of A
        for (int j = row_start + lane; j < row_end; j += WARP_SIZE) {
            int col = A_col_idx[j];
            float val = A_values[j];
            sum += val * __ldg(&B[col * K + n]);
        }

        // sum all partials in warp using tree reduction 
        // NOTE: offset >>= 1 = offset / 2
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) 
            sum += __shfl_down_sync(0xffffffff, sum, offset);

        if (lane == 0)
            C[warp_id * K + n] = alpha * sum + beta * C[warp_id * K + n];
    }
}



// TODO:
// - problem with colescing access
// - buffering to share mem, block tiling on B
// - multi warps per row
// - load balancing
