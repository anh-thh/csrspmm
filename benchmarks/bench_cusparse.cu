#include "helper.cuh"
#include "csr_utils.cuh"
#include "dense_utils.cuh"
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <string>

#define WARMUP 200
#define REPS 1000

int main (int argc, char** argv) {
    int M = 1024;
    int N = 1024;
    int K = 1024;
    float sparsity = 0.7;
    float alpha = 1.0f;
    float beta  = 0.5f;

    // parse args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if      ((arg == "-M") && i+1 < argc) M = std::atoi(argv[++i]);
        else if ((arg == "-N") && i+1 < argc) N = std::atoi(argv[++i]);
        else if ((arg == "-K") && i+1 < argc) K = std::atoi(argv[++i]);
        else if ((arg == "-s") && i+1 < argc) sparsity = std::atof(argv[++i]);
        else if ((arg == "-a") && i+1 < argc) alpha    = std::atof(argv[++i]);
        else if ((arg == "-b") && i+1 < argc) beta     = std::atof(argv[++i]);

        else if (arg == "-h" || arg == "--help") {
            std::cout <<
                "Usage: " << argv[0] << " [options]\n"
                "  -M <int>      number of rows of A (and C)\n"
                "  -N <int>      number of columns of A\n"
                "  -K <int>      number of columns of B (and C)\n"
                "  -s <float>    sparsity (0.0 - 1.0)\n"
                "  -a <float>    alpha scalar\n"
                "  -b <float>    beta scalar\n"
                "Example:\n"
                "  ./bench_csr_cusparse -M 4096 -N 4096 -K 256 -s 0.999\n";
            return 0;
        }

        else {
            std::cerr << "Unknown or incomplete flag: " << arg << std::endl;
            return 1;
        }
    }    
    
    
    bool is_int = false;

    float min_val = -10.0f;
    float max_val = 10.0f;

    float dense2csr_tol = 0.0f;

    // create matrices 
    float* h_A = (float*)malloc(M * N * sizeof(float));
    float* h_B = (float*)malloc(N * K * sizeof(float));
    float* h_C = (float*)malloc(M * K * sizeof(float));
    init_random_dense_matrix(h_A, M, N, min_val, max_val, sparsity, is_int);
    init_random_dense_matrix(h_B, N, K, min_val, max_val, 0.0f, is_int);
    init_random_dense_matrix(h_C, M, K, min_val, max_val, 0.0f, is_int);
    
    CSRMatrix A_csr;
    dense2csr(h_A, M, N, A_csr, dense2csr_tol);


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
    cudaCheck(cudaMalloc(&d_B, N * K * sizeof(float)));
    cudaCheck(cudaMalloc(&d_C, M * K * sizeof(float)));
    cudaCheck(cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_C, h_C, M * K * sizeof(float), cudaMemcpyHostToDevice));

    
    // ----------------------------------------------------------------------
    //      cusparse setup
    // ----------------------------------------------------------------------
    cusparseHandle_t handle;
    cusparseCheck(cusparseCreate(&handle));

    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    cusparseCheck(cusparseCreateCsr(
        &matA,
        M, N, A_csr.nnz,
        (void*)d_A_csr.row_ptr,
        (void*)d_A_csr.col_idx,
        (void*)d_A_csr.values,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F));

    cusparseCheck(cusparseCreateDnMat(
        &matB, N, K, K, (void*)d_B, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    cusparseCheck(cusparseCreateDnMat(
        &matC, M, K, K, (void*)d_C, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    size_t bufferSize = 0;
    void*  dBuffer    = nullptr;
    cusparseCheck(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,   // opA
        CUSPARSE_OPERATION_NON_TRANSPOSE,   // opB
        &alpha,
        matA, matB,
        &beta,
        matC,
        CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        &bufferSize));
    if (bufferSize > 0) cudaCheck(cudaMalloc(&dBuffer, bufferSize));


    // warm up
    for (int i = 0; i < WARMUP; ++i) {
        cusparseCheck(cusparseSpMM(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            matA, matB,
            &beta,
            matC,
            CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT,
            dBuffer));
    }
    cudaCheck(cudaDeviceSynchronize());
   

    // ----------------------------------------------------------------------
    //      Benchmarking
    // ----------------------------------------------------------------------
    cudaEvent_t beg, end;
    cudaCheck(cudaEventCreate(&beg));
    cudaCheck(cudaEventCreate(&end));

    cudaCheck(cudaEventRecord(beg));
    for (int j = 0; j < REPS; j++) {
        cusparseCheck(cusparseSpMM(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            matA, matB,
            &beta,
            matC,
            CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT,
            dBuffer));
        cudaCheck(cudaDeviceSynchronize());
    }

    cudaCheck(cudaEventRecord(end));
    cudaCheck(cudaEventSynchronize(beg));
    cudaCheck(cudaEventSynchronize(end));
 
    float elapsed_time;
    cudaCheck(cudaEventElapsedTime(&elapsed_time, beg, end));
    elapsed_time /= 1000.; // Convert to seconds

    double flops = 2.0 * A_csr.nnz * K;  // NOTE: 2 ops per nonzero × K columns
    printf(
        "cuSPARSE api: avg elapsed time: (%7.6f) s, performance: (%7.2f) GFLOPS. size: [%u×%u×%u]\n",
        elapsed_time / REPS,
        (REPS * flops * 1e-9) / elapsed_time,
        M, N, K);


    // ----------------------------------------------------------------------
    //      Clean up
    // ----------------------------------------------------------------------
    cudaCheck(cudaEventDestroy(beg));
    cudaCheck(cudaEventDestroy(end));

    if (dBuffer) cudaCheck(cudaFree(dBuffer));
    cusparseCheck(cusparseDestroySpMat(matA));
    cusparseCheck(cusparseDestroyDnMat(matB));
    cusparseCheck(cusparseDestroyDnMat(matC));
    cusparseCheck(cusparseDestroy(handle));

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
