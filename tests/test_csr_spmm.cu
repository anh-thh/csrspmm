#include "helper.cuh"
#include <cuda_runtime.h>
#include "csr_utils.cuh"
#include "csr_spmm.cuh"
#include "dense_utils.cuh"
#include "dense_gemm.cuh"
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <iostream>

#define cudaCheck(err) (cudaErrorCheck(err, __FILE__, __LINE__))
#define cublasCheck(err) (cublasErrorCheck(err, __FILE__, __LINE__))


int main(int argc, char** argv) {
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
    float cmp_tol = 1e-5f;

    // Generate A, B and C
    float* h_A = (float*)malloc(M * K * sizeof(float));
    float* h_B = (float*)malloc(K * N * sizeof(float));
    float* h_C = (float*)malloc(M * N * sizeof(float));
    init_random_dense_matrix(h_B, K, N, min_val, max_val, 0.0f, is_int);
    init_random_dense_matrix(h_C, M, N, min_val, max_val, 0.0f, is_int);
    init_random_dense_matrix(h_A, M, K, min_val, max_val, sparsity, is_int);
    
    CSRMatrix A_csr;
    dense2csr(h_A, M, K, A_csr, dense2csr_tol);

    // // print
    // std::cout << "A:" << std::endl;
    // print_dense_matrix(h_A, M, K);
    //
    // std::cout << "A:" << std::endl;
    // print_csr_matrix(A_csr);
    //
    // std::cout << "B:" << std::endl;
    // print_dense_matrix(h_B, K, N);

    // Allocate device memory
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

    // Launch kernel
    dim3 gridDim(ROUND_UP_TO_NEAREST(M, 32));
    dim3 blockDim(32);   

    csr_spmm_naive<<<gridDim, blockDim>>>(
        M, N, K,
        alpha, beta,
        d_A_csr.values, d_A_csr.col_idx, d_A_csr.row_ptr,
        d_B, d_C);
   
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    

    // Copy result back to host
    float* h_C_gpu = (float*)malloc(M * N * sizeof(float));
    cudaCheck(cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // cpu reference
    float* h_C_ref = (float*)malloc(M * N * sizeof(float));
    memcpy(h_C_ref, h_C, M * N * sizeof(float));
    dense_gemm_cpu(M, N, K, alpha, h_A, h_B, beta, h_C_ref);

    // std::cout << "C_gpu:" << std::endl;
    // print_dense_matrix(h_C_gpu, M, N);
    // std::cout << "C_ref:" << std::endl;
    // print_dense_matrix(h_C_ref, M, N);
    
    // compare
    bool correct = compare_dense_matrices(h_C_ref, h_C_gpu, M, N, cmp_tol);
    if (correct) {
        std::cout << "Pass: SPMM CSR kernel result is correct!" << std::endl;
    } else {
        std::cout << "Error: SPMM CSR kernel result is incorrect!" << std::endl;
    }

    // free memory
    free(h_A);
    free(A_csr.row_ptr);
    free(A_csr.col_idx);
    free(A_csr.values);
    free(h_B);
    free(h_C);
    free(h_C_gpu);
    free(h_C_ref);
    cudaCheck(cudaFree(d_A_csr.row_ptr));
    cudaCheck(cudaFree(d_A_csr.col_idx));
    cudaCheck(cudaFree(d_A_csr.values));
    cudaCheck(cudaFree(d_B));
    cudaCheck(cudaFree(d_C));
    
    return 0;
}
