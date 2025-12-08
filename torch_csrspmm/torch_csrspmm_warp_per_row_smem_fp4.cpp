#include <torch/extension.h>
#include <iostream>

#include "csrspmm/matrix.h"
#include "csrspmm/spmm_launcher.cuh"

using torch::Tensor;

torch::Tensor csrspmm_warp_per_row_smem_fp4_forward(
    Tensor A_row_ptr,
    Tensor A_col_idx,
    Tensor A_values,
    Tensor B,
    float alpha,
    float beta
) {
    // ---- Basic checks ----
    TORCH_CHECK(A_row_ptr.is_cuda(), "A_row_ptr must be CUDA");
    TORCH_CHECK(A_col_idx.is_cuda(), "A_col_idx must be CUDA");
    TORCH_CHECK(A_values.is_cuda(),  "A_values must be CUDA");
    TORCH_CHECK(B.is_cuda(),         "B must be CUDA");

    TORCH_CHECK(A_row_ptr.dtype() == torch::kInt32);
    TORCH_CHECK(A_col_idx.dtype()  == torch::kInt32);
    TORCH_CHECK(A_values.dtype()   == torch::kFloat32);
    TORCH_CHECK(B.dtype()          == torch::kFloat32);

    TORCH_CHECK(B.dim() == 2, "B must be 2D [K, N]");

    int M = A_row_ptr.size(0) - 1;
    int K = B.size(0);
    int N = B.size(1);

    TORCH_CHECK(A_values.size(0) == A_col_idx.size(0),
                "nnz mismatch: A_values and A_col_idx sizes differ");

    // ---- FP4 requirement ----
    TORCH_CHECK(K % 4 == 0,
        "FP4 kernel requires K divisible by 4, got K = ", K);

    // ---- Allocate output ----
    Tensor C = torch::zeros({M, N}, B.options());

    TORCH_CHECK(B.is_contiguous(), "B must be contiguous for FP4 kernel");
    TORCH_CHECK(C.is_contiguous(), "C must be contiguous");

    // ---- Compute max_row_nnz using non-blocking CPU transfer ----
    auto row_cpu = A_row_ptr.to(torch::kCPU, /*non_blocking=*/true);
    const int* h_row = row_cpu.data_ptr<int>();

    int max_row_nnz = 0;
    for (int i = 0; i < M; ++i) {
        int nnz_row = h_row[i + 1] - h_row[i];
        if (nnz_row > max_row_nnz) max_row_nnz = nnz_row;
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

    // ---- Dense matrices ----
    csrspmm::DenseMatrix dB{B.data_ptr<float>(), B.size(0), B.size(1)};
    csrspmm::DenseMatrix dC{C.data_ptr<float>(), C.size(0), C.size(1)};

    // ---- Launch kernel ----
    csrspmm::launch_warp_per_row_smem_fp4(dA, dB, dC, alpha, beta);

    return C;
}
