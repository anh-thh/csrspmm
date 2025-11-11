#include "helper.cuh"
#include "csr_utils.cuh"
#include "csr_spmm.cuh"
#include "dense_utils.cuh"

enum Algo {
    cuSPARSELt = 0,
    naive,
    warp,
    numAlgos
};

#define ALGO naive
#define REPS 100

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

    default:
        printf("Invalid algorithm: %d\n", algo);
        exit(EXIT_FAILURE);
    }

    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError());
}


int main (int argc, char** argv) {
    // ----------------------------------------------------------------------
    //      Setup
    // ----------------------------------------------------------------------
    int M = 1024;
    int N = 1024;
    int K = 1024;
    float sparsity = 0.7;
    bool is_int = false;

    float min_val = -10.0f;
    float max_val = 10.0f;

    float alpha = 1.0f;
    float beta  = 0.5f;
    
    float dense2csr_tol = 0.0f;

    // create matrices 
    float* h_A = (float*)malloc(M * K * sizeof(float));
    float* h_B = (float*)malloc(K * N * sizeof(float));
    float* h_C = (float*)malloc(M * N * sizeof(float));
    init_random_dense_matrix(h_B, K, N, min_val, max_val, 0.0f, is_int);
    init_random_dense_matrix(h_C, M, N, min_val, max_val, 0.0f, is_int);
    init_random_dense_matrix(h_A, M, K, min_val, max_val, sparsity, is_int);
   
    CSRMatrix A_csr;
    dense2csr(h_A, M, K, A_csr, dense2csr_tol);


    CSRMatrix d_A_csr;
    cudaCheck(cudaMalloc(&d_A_csr.values,  A_csr.nnz * sizeof(float)));
    cudaCheck(cudaMalloc(&d_A_csr.col_idx, A_csr.nnz * sizeof(int)));
    cudaCheck(cudaMalloc(&d_A_csr.row_ptr, (A_csr.num_rows + 1) * sizeof(int)));
    cudaCheck(cudaMemcpy(d_A_csr.values,  A_csr.values,  
                         A_csr.nnz * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_A_csr.col_idx, A_csr.col_idx, 
                         A_csr.nnz * sizeof(int),   cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_A_csr.row_ptr, A_csr.row_ptr, 
                        (A_csr.num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));

    float *d_B, *d_C;
    cudaCheck(cudaMalloc(&d_B, K * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_C, M * N * sizeof(float)));
    cudaCheck(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));


    // warm up
    for (int i = 0; i < 10; ++i) {
        run_csr_spmm(ALGO, M, N, K,
                     alpha, beta, 
                     d_A_csr.values,
                     d_A_csr.col_idx,
                     d_A_csr.row_ptr,
                     d_B, d_C);
    }
   

    // ----------------------------------------------------------------------
    //      Benchmarking
    // ----------------------------------------------------------------------
    cudaEvent_t beg, end;
    cudaCheck(cudaEventCreate(&beg));
    cudaCheck(cudaEventCreate(&end));

    cudaEventRecord(beg);
    for (int j = 0; j < REPS; j++)
    {
        // runAlgo(ALGO, handle, m, n, k, alpha, dA, dB, beta, dC);
        run_csr_spmm(ALGO, M, N, K,
                     alpha, beta, 
                     d_A_csr.values,
                     d_A_csr.col_idx,
                     d_A_csr.row_ptr,
                     d_B, d_C);
    }

    cudaCheck(cudaEventRecord(end));
    cudaCheck(cudaEventSynchronize(beg));
    cudaCheck(cudaEventSynchronize(end));
 
    float elapsed_time;
    cudaCheck(cudaEventElapsedTime(&elapsed_time, beg, end));
    elapsed_time /= 1000.; // Convert to seconds

    double flops = 2.0 * A_csr.nnz * N;  // NOTE: 2 ops per nonzero × N columns
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.2f) GFLOPS. size: [%u×%u×%u]\n",
        elapsed_time / REPS,
        (REPS * flops * 1e-9) / elapsed_time,
        M, N, K);


    // ----------------------------------------------------------------------
    //      Clean up
    // ----------------------------------------------------------------------
    free(h_A);
    free(A_csr.row_ptr);
    free(A_csr.col_idx);
    free(A_csr.values);
    free(h_B);
    free(h_C);
    cudaCheck(cudaFree(d_A_csr.row_ptr));
    cudaCheck(cudaFree(d_A_csr.col_idx));
    cudaCheck(cudaFree(d_A_csr.values));
    cudaCheck(cudaFree(d_B));
    cudaCheck(cudaFree(d_C));
    return 0;
}
