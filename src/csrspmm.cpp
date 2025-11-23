#include <cctype>
#include <iostream>
#include <csrspmm/csrspmm.h>
#include <csrspmm/kernels.h>
#include <csrspmm/config.h>
#include <csrspmm/error_check.h>
#include "csrspmm/spmm_launcher.cuh"

namespace csrspmm {



Algorithm parse_algorithm(const std::string& name)
{
    if (name == "Naive")                return Algorithm::Naive;
    if (name == "WarpPerRow")           return Algorithm::WarpPerRow;
    if (name == "WarpPerRowVec4")       return Algorithm::WarpPerRowVec4;
    if (name == "WarpPerRowSharemem")   return Algorithm::WarpPerRowSharemem;

    std::cerr << "[csrspmm] Unknown algorithm: " << name
              << ", using Naive\n";
    return Algorithm::Naive;
}


void spmm(const CSRMatrix& A,
          const DenseMatrix& B,
          DenseMatrix& C,
          float alpha, float beta,
          Algorithm algo)
{
    // Auto-selection logic
    if (algo == Algorithm::Naive) {
        if (A.max_row_nnz < 64)
            algo = Algorithm::Naive;
        else
            algo = Algorithm::Naive;
    }

    switch (algo)
    {
        case Algorithm::Naive:
            launch_naive(A, B, C, alpha, beta);
            break;

        case Algorithm::WarpPerRow:
            launch_warp_per_row(A, B, C, alpha, beta);
            break;

        case Algorithm::WarpPerRowVec4:
            launch_warp_per_row_vec4(A, B, C, alpha, beta);
            break;

        case Algorithm::WarpPerRowSharemem:
            launch_warp_per_row_sharemem(A, B, C, alpha, beta);
            break;

        default:
            printf("[csrspmm::spmm] Invalid Algorithm.\n");
            return;
    }

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
}

} // namespace csrspmm

