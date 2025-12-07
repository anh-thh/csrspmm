#include <string>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cusparse.h>

#include <csrspmm/matrix.h>
#include <csrspmm/csr_utils.h>
#include <csrspmm/dense_utils.h>
#include <csrspmm/csrspmm.h>
#include <csrspmm/error_check.h>

#include "helper.cuh"

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
    
    
    // csrspmm::Algorithm algo = csrspmm::parse_algorithm(algo_str);
    bool is_int = false;

    float min_val = -10.0f;
    float max_val = 10.0f;

    float dense2csr_tol = 0.0f;

    // Generate A, B and C
    csrspmm::DenseMatrix hA, hB, hC;

    csrspmm::dense_alloc_host(hA, M, N);
    csrspmm::dense_alloc_host(hB, N, K);
    csrspmm::dense_alloc_host(hC, M, K);
    
    csrspmm::dense_init_random(hA, min_val, max_val, sparsity, is_int);
    csrspmm::dense_init_random(hB, min_val, max_val, 0.0f, is_int);
    csrspmm::dense_init_random(hC, min_val, max_val, 0.0f, is_int);

    csrspmm::CSRMatrix hA_CSR;
    csrspmm::dense2csr(hA, hA_CSR, dense2csr_tol);

    // Allocate device memory
    csrspmm::CSRMatrix dA_CSR;
    csrspmm::csr_host2device(hA_CSR, dA_CSR);

    csrspmm::DenseMatrix dB, dC;
    csrspmm::dense_host2device(hB, dB);
    csrspmm::dense_host2device(hC, dC);

    // csrspmm::spmm(dA_CSR, dB, dC, alpha, beta, algo);

    
    // ----------------------------------------------------------------------
    //      cusparse setup
    // ----------------------------------------------------------------------
    cusparseHandle_t handle;
    cusparseCheck(cusparseCreate(&handle));

    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    cusparseCheck(cusparseCreateCsr(
        &matA,
        M, N, dA_CSR.nnz,
        (void*)dA_CSR.row_ptr,
        (void*)dA_CSR.col_idx,
        (void*)dA_CSR.values,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F));

    cusparseCheck(cusparseCreateDnMat(
        &matB, N, K, K, (void*)dB.data, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    cusparseCheck(cusparseCreateDnMat(
        &matC, M, K, K, (void*)dC.data, CUDA_R_32F, CUSPARSE_ORDER_ROW));

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

    double flops = 2.0 * dA_CSR.nnz * K;  // NOTE: 2 ops per nonzero × K columns
    printf(
        "cuSPARSE api: avg elapsed time: (%7.6f) s, performance: (%7.2f) GFLOPS. size: [%u x %u x %u]\n",
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

    csrspmm::dense_free_host(hA);
    csrspmm::dense_free_host(hB);
    csrspmm::dense_free_host(hC);
    csrspmm::csr_free_host(hA_CSR);
    csrspmm::csr_free_device(dA_CSR);
    csrspmm::dense_free_device(dB);
    csrspmm::dense_free_device(dC);
    return 0;
}
