#include <string>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iostream>

#include <csrspmm/matrix.h>
#include <csrspmm/csr_utils.h>
#include <csrspmm/dense_utils.h>
#include <csrspmm/csrspmm.h>
#include <csrspmm/error_check.h>

#include "helper.cuh"

#define WARMUP 200
#define  REPS 1000

int main (int argc, char** argv) {
    int M = 1024;
    int N = 1024;
    int K = 1024;
    float sparsity = 0.7;
    float alpha = 1.0f;
    float beta  = 0.5f;
    std::string algo_str = "Naive";  // default algorithm

    // parse args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if      ((arg == "-M") && i+1 < argc) M = std::atoi(argv[++i]);
        else if ((arg == "-N") && i+1 < argc) N = std::atoi(argv[++i]);
        else if ((arg == "-K") && i+1 < argc) K = std::atoi(argv[++i]);
        else if ((arg == "-s") && i+1 < argc) sparsity = std::atof(argv[++i]);
        else if ((arg == "-a") && i+1 < argc) alpha    = std::atof(argv[++i]);
        else if ((arg == "-b") && i+1 < argc) beta     = std::atof(argv[++i]);
        else if ((arg == "-algo") && i+1 < argc) algo_str = argv[++i];

        else if (arg == "-h" || arg == "--help") {
            std::cout <<
                "Usage: " << argv[0] << " [options]\n"
                "  -M <int>      number of rows of A (and C)\n"
                "  -N <int>      number of columns of A\n"
                "  -K <int>      number of columns of B (and C)\n"
                "  -s <float>    sparsity (0.0 - 1.0)\n"
                "  -a <float>    alpha scalar\n"
                "  -b <float>    beta scalar\n"
                "  -algo <str>   naive | warp_per_row | adaptive\n"
                "Example:\n"
                "  ./bench_csr_spmm -M 4096 -N 4096 -K 256 -s 0.999 -algo WarpPerRow\n";
            return 0;
        }

        else {
            std::cerr << "Unknown or incomplete flag: " << arg << std::endl;
            return 1;
        }
    }    


    csrspmm::Algorithm algo = csrspmm::parse_algorithm(algo_str);
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

    csrspmm::spmm(dA_CSR, dB, dC, alpha, beta, algo);


    // warm up
    for (int i = 0; i < WARMUP; ++i) {
        csrspmm::spmm(dA_CSR, dB, dC, alpha, beta, algo);
    }
    cudaCheck(cudaDeviceSynchronize());
   
    // ----------------------------------------------------------------------
    //      Benchmarking
    // ----------------------------------------------------------------------
    cudaEvent_t beg, end;
    cudaCheck(cudaEventCreate(&beg));
    cudaCheck(cudaEventCreate(&end));

    cudaCheck(cudaEventRecord(beg));
    for (int j = 0; j < REPS; j++)
    {
        csrspmm::spmm(dA_CSR, dB, dC, alpha, beta, algo);
    }

    cudaCheck(cudaEventRecord(end));
    cudaCheck(cudaEventSynchronize(beg));
    cudaCheck(cudaEventSynchronize(end));
 
    float elapsed_time;
    cudaCheck(cudaEventElapsedTime(&elapsed_time, beg, end));
    elapsed_time /= 1000.; // Convert to seconds

    double flops = 2.0 * dA_CSR.nnz * K;  // NOTE: 2 ops per nonzero × K columns
    printf(
        "%s krnl: avg elapsed time: (%7.6f) s, performance: (%7.2f) GFLOPS. size: [%u x %u x %u]\n",
        algo_str.c_str(),
        elapsed_time / REPS,
        (REPS * flops * 1e-9) / elapsed_time,
        M, N, K);


    // ----------------------------------------------------------------------
    //      Clean up
    // ----------------------------------------------------------------------
    csrspmm::dense_free_host(hA);
    csrspmm::dense_free_host(hB);
    csrspmm::dense_free_host(hC);
    csrspmm::csr_free_host(hA_CSR);
    csrspmm::csr_free_device(dA_CSR);
    csrspmm::dense_free_device(dB);
    csrspmm::dense_free_device(dC);

    return 0;
}
