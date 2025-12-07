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
    if (name == "Naive")             return Algorithm::Naive;
    if (name == "WarpPerRow")        return Algorithm::WarpPerRow;
    if (name == "WarpPerRowFp4")     return Algorithm::WarpPerRowFp4;
    if (name == "WarpPerRowSmem")    return Algorithm::WarpPerRowSmem;
    if (name == "WarpPerRowSmemFp4") return Algorithm::WarpPerRowSmemFp4;
    if (name == "NaiveShared")       return Algorithm::NaiveShared;

    std::cerr << "[csrspmm] Unknown algorithm: " << name
              << ", using Naive\n";
    return Algorithm::Naive;
}

void spmm(const CSRMatrix& A,
          const DenseMatrix& B,
          DenseMatrix& C,
          float alpha,
          float beta,
          Algorithm algo)
{
    switch (algo)
    {
        case Algorithm::Naive:
            launch_naive(A, B, C, alpha, beta);
            break;

        case Algorithm::WarpPerRow:
            launch_warp_per_row(A, B, C, alpha, beta);
            break;

        case Algorithm::WarpPerRowFp4:
            launch_warp_per_row_fp4(A, B, C, alpha, beta);
            break;

        case Algorithm::WarpPerRowSmem:
            launch_warp_per_row_smem(A, B, C, alpha, beta);
            break;

        case Algorithm::WarpPerRowSmemFp4:
            launch_warp_per_row_smem_fp4(A, B, C, alpha, beta);
            break;

        case Algorithm::NaiveShared:
            launch_naive_shared(A, B, C, alpha, beta);
            break;

        default:
            printf("[csrspmm::spmm] Invalid Algorithm.\n");
            return;
    }

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
}

} // namespace csrspmm
