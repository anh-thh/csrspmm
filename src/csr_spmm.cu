#include <cuda_runtime.h>
#include <vector_types.h>
#include "csr_spmm.cuh"
#include "csr_utils.cuh"
#include "helper.cuh"


Algo parse_csr_algo(const std::string& name) {
    if (name == "cusparse")             return cusparse;
    if (name == "naive")                return naive;
    if (name == "warp_per_row")         return warp_per_row;

    std::cerr << "Error: unknown algorithm name '" << name << "'\n";
    std::exit(1);
}


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
    case cusparse: {
        // Placeholder for cuSPARSELt
        break;
    }

    case naive: {
        dim3 block(32, 32);
        dim3 grid((K + block.x - 1) / block.x,
                  (M + block.y - 1) / block.y);

        csr_spmm_naive<<<grid, block>>>(
            M, N, K, alpha, beta,
            A_values, A_col_idx, A_row_ptr,
            B, C
        );
        break;
    }

    case warp_per_row: {
        const int threads_per_block = 128;   // 4 warps
        const int warps_per_block   = threads_per_block / 32;

        int num_blocks = (M + warps_per_block - 1) / warps_per_block;

        csr_spmm_warp_per_row<<<num_blocks, threads_per_block>>>(
            M, N, K,
            alpha, beta,
            A_values,
            A_col_idx,
            A_row_ptr,
            B,
            C
        );
        break;
    }

    default:
        printf("Invalid algorithm: %d\n", algo);
        exit(EXIT_FAILURE);
    }

    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError());
}


/**
 * Each thread computes a single element C[row, col], finer-grained parallelism 
 * across both dimensions
 */
__global__ 
void csr_spmm_naive(
    int M, int N, int K,
    float alpha,
    float beta,
    const float* __restrict__ A_values,
    const int* __restrict__ A_col_idx,
    const int* __restrict__ A_row_ptr,
    const float* __restrict__ B,
    float* __restrict__ C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // row in C
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // column in C

    if (row >= M || col >= K) return;

    int row_start = A_row_ptr[row];
    int row_end   = A_row_ptr[row + 1];

    float sum = 0.0f;
    for (int j = row_start; j < row_end; ++j) {
        int A_col   = A_col_idx[j];
        float A_val = A_values[j];
        sum        += A_val * __ldg(&B[A_col * K + col]);
    }
    
    // C[row * K + col] = alpha * sum + beta * C[row * K + col];
    C[row * K + col] = alpha * sum + (beta != 0.0f ? beta * C[row * K + col] : 0.0f);
}



#define WARP_SIZE 32
/**
 * Each warp loads one sparse row of A and processes WARP_SIZE (32) columns of B.
 * Lane k (of the warp) computes output columns {k, k+32, k+64, ...}, producing C[row, lane::32].
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
    const int lane            = threadIdx.x % WARP_SIZE;

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



// TODO:
// - problem with colescing access
// - buffering to share mem, block tiling on B
// - multi warps per row
// - load balancing
