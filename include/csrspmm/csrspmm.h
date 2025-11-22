#pragma once
#include <string>
#include <csrspmm/matrix.h>
#include <csrspmm/config.h>
#include <csrspmm/csr_utils.h>  
#include <csrspmm/kernels.h>  

namespace csrspmm {

enum class Algorithm {
    Naive,
    WarpPerRow,
    WarpPerRowVec4,
    WarpPerRowSharemem,
};

Algorithm parse_algorithm(const std::string& name);

/**
 * High-level CSR SpMM API
 *
 * A: CSR (M x N)
 * B: Dense (N x K)
 * C: Dense (N x K)
 * alpha, beta: float
 * Algo: choose kernel variant
 */
void spmm(const CSRMatrix& A,
          const DenseMatrix& B,
          DenseMatrix& C,
          float alpha, float beta,
          Algorithm algo = Algorithm::Naive);

} 


