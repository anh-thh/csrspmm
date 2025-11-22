#include "helper.cuh"
#include "csr_utils.cuh"
#include "csr_spmm.cuh"
#include "dense_utils.cuh"
#include <string>

#define REPS 500


int main (int argc, char** argv) {
    int M = 1024;
    int N = 1024;
    int K = 1024;
    float sparsity = 0.7;
    float alpha = 1.0f;
    float beta  = 0.5f;
    std::string algo_str = "naive";  // default algorithm

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
                "  ./bench_csr_spmm -M 4096 -N 4096 -K 256 -s 0.999 -algo warp_per_row\n";
            return 0;
        }

        else {
            std::cerr << "Unknown or incomplete flag: " << arg << std::endl;
            return 1;
        }
    }    
    
    Algo algo = parse_csr_algo(algo_str);
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

    int A_max_row_nnz = 0;
    for (int i = 0; i < M; i++) {
        int nnz = A_csr.row_ptr[i+1] - A_csr.row_ptr[i];
        A_max_row_nnz = std::max(A_max_row_nnz, nnz);
    }

    // warm up
    for (int i = 0; i < 10; ++i) {
        run_csr_spmm(algo, M, N, K,
                     alpha, beta, 
                     d_A_csr.values,
                     d_A_csr.col_idx,
                     d_A_csr.row_ptr,
                     A_max_row_nnz,
                     d_B, d_C);
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
        // runAlgo(algo, handle, m, n, k, alpha, dA, dB, beta, dC);
        run_csr_spmm(algo, M, N, K,
                     alpha, beta, 
                     d_A_csr.values,
                     d_A_csr.col_idx,
                     d_A_csr.row_ptr,
                     A_max_row_nnz,
                     d_B, d_C);
    }

    cudaCheck(cudaEventRecord(end));
    cudaCheck(cudaEventSynchronize(beg));
    cudaCheck(cudaEventSynchronize(end));
 
    float elapsed_time;
    cudaCheck(cudaEventElapsedTime(&elapsed_time, beg, end));
    elapsed_time /= 1000.; // Convert to seconds

    double flops = 2.0 * A_csr.nnz * K;  // NOTE: 2 ops per nonzero × K columns
    printf(
        "%s krnl: avg elapsed time: (%7.6f) s, performance: (%7.2f) GFLOPS. size: [%u×%u×%u]\n",
        algo_str.c_str(),
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
