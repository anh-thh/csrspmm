#include <cuda_runtime.h>
#include "csr_spmm.cuh"
#include "csr_utils.cuh"
#include "helper.cuh"


Algo parse_csr_algo(const std::string& name) {
    if (name == "cuSPARSELt")   return cuSPARSELt;
    if (name == "naive")        return naive;

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
    case cuSPARSELt: {
        // Placeholder for cuSPARSELt
        break;
    }

    case naive: {
        const int block_size = 32;
        const dim3 gridDim(ROUND_UP_TO_NEAREST(K, block_size), ROUND_UP_TO_NEAREST(M, block_size));
        const dim3 blockDim(block_size, block_size);

        csr_spmm_naive<<<gridDim, blockDim>>>(M, N, K, 
                                              alpha, beta,
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




/**
 * Each thread computes a single element C[row, col], finer-grained parallelism 
 * across both dimensions
 */
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




// TODO:
// - problem with colescing access
// - buffering to share mem, block tiling on B
// - multi warps per row
// - load balancing
