#include <torch/extension.h>
#include <iostream>

#include "csrspmm/matrix.h"
#include "csrspmm/csr_utils.h"
#include "csrspmm/dense_utils.h"
#include "csrspmm/spmm_launcher.cuh"

using torch::Tensor;

torch::Tensor csrspmm_warp_per_row_smem_forward(
    Tensor A_row_ptr,   // int32, size = M+1
    Tensor A_col_idx,   // int32, size = nnz
    Tensor A_values,    // float32, size = nnz
    Tensor B,           // float32, [K x N]
    float alpha,
    float beta
) {
    // ---- Basic checks ----
    TORCH_CHECK(A_row_ptr.is_cuda(),   "A_row_ptr must be CUDA");
    TORCH_CHECK(A_col_idx.is_cuda(),   "A_col_idx must be CUDA");
    TORCH_CHECK(A_values.is_cuda(),    "A_values must be CUDA");
    TORCH_CHECK(B.is_cuda(),           "B must be CUDA");

    TORCH_CHECK(A_row_ptr.dtype() == torch::kInt32,  "A_row_ptr must be int32");
    TORCH_CHECK(A_col_idx.dtype()  == torch::kInt32, "A_col_idx must be int32");
    TORCH_CHECK(A_values.dtype()   == torch::kFloat32, "A_values must be float32");
    TORCH_CHECK(B.dtype()          == torch::kFloat32, "B must be float32");

    int M = A_row_ptr.size(0) - 1;  // rows of A / C
    int K = B.size(0);              // cols of A
    int N = B.size(1);              // cols of B / C

    // ---- Allocate output ----
    Tensor C = torch::zeros({M, N}, B.options());

    // ---- Compute max_row_nnz on CPU from crow ----
    Tensor row_cpu = A_row_ptr.to(torch::kCPU);
    const int* h_row = row_cpu.data_ptr<int>();

    int max_row_nnz = 0;
    for (int i = 0; i < M; ++i) {
        int nnz_row = h_row[i + 1] - h_row[i];
        if (nnz_row > max_row_nnz) {
            max_row_nnz = nnz_row;
        }
    }

    // ---- Fill CSRMatrix ----
    csrspmm::CSRMatrix dA;
    dA.height      = M;
    dA.width       = K;
    dA.nnz         = A_values.size(0);
    dA.row_ptr     = A_row_ptr.data_ptr<int>();
    dA.col_idx     = A_col_idx.data_ptr<int>();
    dA.values      = A_values.data_ptr<float>();
    dA.max_row_nnz = max_row_nnz;

    // ---- Dense B ----
    csrspmm::DenseMatrix dB;
    dB.height = B.size(0);
    dB.width  = B.size(1);
    dB.data   = B.data_ptr<float>();

    // ---- Dense C ----
    csrspmm::DenseMatrix dC;
    dC.height = C.size(0);
    dC.width  = C.size(1);
    dC.data   = C.data_ptr<float>();

    // ---- Launch kernel ----
    csrspmm::launch_warp_per_row_smem(dA, dB, dC, alpha, beta);

    return C;
}
