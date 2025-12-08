#include <torch/extension.h>
#include <iostream>

#include "csrspmm/matrix.h"
#include "csrspmm/csr_utils.h"
#include "csrspmm/dense_utils.h"
#include "csrspmm/spmm_launcher.cuh"

using torch::Tensor;

Tensor csrspmm_naive_shared_forward(
    Tensor A_row_ptr,   // int32 row_ptr
    Tensor A_col_idx,   // int32 col indices
    Tensor A_values,    // float32 values
    Tensor B,           // float32 dense matrix (N x K)
    float alpha,
    float beta
) {
    // ---- Input validation ----
    TORCH_CHECK(A_row_ptr.is_cuda(),  "A_row_ptr must be a CUDA tensor");
    TORCH_CHECK(A_col_idx.is_cuda(),  "A_col_idx must be a CUDA tensor");
    TORCH_CHECK(A_values.is_cuda(),   "A_values must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(),          "B must be a CUDA tensor");

    TORCH_CHECK(A_row_ptr.dtype() == torch::kInt32,  "A_row_ptr must be int32");
    TORCH_CHECK(A_col_idx.dtype()  == torch::kInt32, "A_col_idx must be int32");
    TORCH_CHECK(A_values.dtype()   == torch::kFloat32, "A_values must be float32");

    // A: M x N (CSR), B: N x K
    int M = A_row_ptr.size(0) - 1;
    int K = B.size(1);   // output width
    int N = B.size(0);   // A.width

    // ---- Allocate output tensor ----
    Tensor C = torch::zeros({M, K}, B.options());

    // ---- Build CSRMatrix for A ----
    csrspmm::CSRMatrix dA;
    dA.height  = M;
    dA.width   = N;
    dA.nnz     = static_cast<int>(A_values.size(0));
    dA.row_ptr = A_row_ptr.data_ptr<int>();
    dA.col_idx = A_col_idx.data_ptr<int>();
    dA.values  = A_values.data_ptr<float>();

    // ---- Dense matrix B ----
    csrspmm::DenseMatrix dB;
    dB.height = B.size(0);   // N
    dB.width  = B.size(1);   // K
    dB.data   = B.data_ptr<float>();

    // ---- Dense matrix C ----
    csrspmm::DenseMatrix dC;
    dC.height = C.size(0);   // M
    dC.width  = C.size(1);   // K
    dC.data   = C.data_ptr<float>();

    // Launch shared-memory variant
    launch_naive_shared(dA, dB, dC, alpha, beta);

    return C;
}
