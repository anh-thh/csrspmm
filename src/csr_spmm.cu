#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector>
#include "csr_spmm.cuh"
#include "csr_utils.cuh"
#include "helper.cuh"


Algo parse_csr_algo(const std::string& name) {
    if (name == "cusparse")                     return cusparse;
    if (name == "naive")                        return naive;
    if (name == "warp_per_row")                 return warp_per_row;
    if (name == "warp_per_row_sharemem")        return warp_per_row_sharemem;

    std::cerr << "Error: unknown algorithm name '" << name << "'\n";
    std::exit(1);
}


/*
 * Sparse x Dense Matrix Multiplication 
 * Top-level function (assume input matrices ready on CUDA device)
 * A[M, N] x B[N, K] => C[M, K]
 */
void run_csr_spmm(Algo algo,
                  int M, int N, int K,
                  float alpha, float beta,
                  float* A_values,
                  int* A_col_idx,
                  int* A_row_ptr,
                  int A_max_row_nnz,
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
        const int warps_per_block = 4;     // 4 warp
        const int threads_per_block = warps_per_block * WARP_SIZE;

        int num_blocks = (M + warps_per_block - 1) / warps_per_block;

        csr_spmm_warp_per_row<<<num_blocks, threads_per_block>>>(
            M, N, K,
            alpha, beta,
            A_values,
            A_col_idx,
            A_row_ptr,
            B, C
        );
        break;
    }

    case warp_per_row_sharemem: {
        const int warps_per_block = 4;     // 4 warp
        const int threads_per_block = warps_per_block * WARP_SIZE;

        size_t shmem_size = (size_t)warps_per_block *
                            (size_t)A_max_row_nnz *
                            (sizeof(float) + sizeof(int));

        int num_blocks = (M + warps_per_block - 1) / warps_per_block;

        csr_spmm_warp_per_row_sharemem<<<
            num_blocks,
            threads_per_block,
            shmem_size>>>(
            M, N, K,
            alpha, beta,
            A_values,
            A_col_idx,
            A_row_ptr,
            A_max_row_nnz,
            B, C
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
 * Each thread load full row of A and full column of B => produce one output entry of C
 * finer-grained parallelism across both dimensions
 * require 2D block and grid to run
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
    const int lane            = threadIdx.x % WARP_SIZE;

    int row = blockIdx.x * warps_per_block + warp_in_block;
    if (row >= M) return;

    int row_start = A_row_ptr[row];
    int row_end   = A_row_ptr[row + 1];

    const int K_vec = K / 4;     // # of full float4 segments
    const int K_tail = K % 4;     // tail columns

    float4* Crow4 = reinterpret_cast<float4*>(C + row * K);

    // Each lane computes vectors at indices lane, lane+WARP_SIZE, lane+2*WARP_SIZE, ...
    // Each vector = 4 output columns.
    main_vetorized_loop: for (int col4 = lane; col4 < K_vec; col4 += WARP_SIZE) {
        float4 acc = make_float4(0,0,0,0);

        for (int j = row_start; j < row_end; j++) {
            float a = __ldg(&A_values[j]);
            int   c = __ldg(&A_col_idx[j]);

            const float4* Brow4 = reinterpret_cast<const float4*>(B + c * K);
            float4 b = __ldg(&Brow4[col4]);

            acc.x += a * b.x;
            acc.y += a * b.y;
            acc.z += a * b.z;
            acc.w += a * b.w;
        }

        float4 old = make_float4(0,0,0,0);
        if (beta != 0.0f)
            old = Crow4[col4];

        Crow4[col4] = make_float4(
            alpha*acc.x + beta*old.x,
            alpha*acc.y + beta*old.y,
            alpha*acc.z + beta*old.z,
            alpha*acc.w + beta*old.w
        );
    }

    if (K_tail != 0) {
        int base_col = K_vec * 4;     // tail column

        for (int col = base_col + lane; col < K; col += WARP_SIZE) {
            float sum = 0.f;

            for (int j = row_start; j < row_end; j++) {
                float a = __ldg(&A_values[j]);
                int   c = __ldg(&A_col_idx[j]);
                sum += a * __ldg(&B[c*K + col]);
            }

            float old = (beta != 0.0f ? C[row*K + col] : 0.0f);
            C[row*K + col] = alpha * sum + beta * old;
        }
    }
}


/**
 * Since all threads in a warp reuse the same A row, we stage its values and
 * column indices in per-warp shared memory to reduce global memory traffic.
 * All remaining logic follows the warp_per_row implementation.
 */
__global__
void csr_spmm_warp_per_row_sharemem(
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
    const int lane            = threadIdx.x % WARP_SIZE;

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





// TODO:
// - problem with colescing access
// - buffering to share mem, block tiling on B
// - multi warps per row
// - load balancing
