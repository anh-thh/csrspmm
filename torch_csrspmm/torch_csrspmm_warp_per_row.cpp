#include <torch/extension.h>
#include "csrspmm/matrix.h"
#include "csrspmm/csr_utils.h"
#include "csrspmm/dense_utils.h"
#include "csrspmm/spmm_launcher.cuh"

using torch::Tensor;

torch::Tensor csrspmm_warp_per_row_forward(
    Tensor A_row_ptr,
    Tensor A_col_idx,
    Tensor A_values,
    Tensor B,
    float alpha,
    float beta
) {
    TORCH_CHECK(A_row_ptr.is_cuda(),   "A_row_ptr must be a CUDA tensor");
    TORCH_CHECK(A_col_idx.is_cuda(),    "A_col_idx must be a CUDA tensor");
    TORCH_CHECK(A_values.is_cuda(), "A_values must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(),      "B must be a CUDA tensor");

    TORCH_CHECK(A_row_ptr.dtype() == torch::kInt32,  "A_row_ptr must be int32");
    TORCH_CHECK(A_col_idx.dtype()  == torch::kInt32,  "A_col_idx must be int32");
    TORCH_CHECK(A_values.dtype() == torch::kFloat32, "A_values must be float32");

    int M = A_row_ptr.size(0) - 1;
    int K = B.size(0);
    int N = B.size(1);

    Tensor C = (beta == 0.0f)
        ? torch::empty({M, N}, B.options())
        : torch::zeros({M, N}, B.options());

    csrspmm::CSRMatrix dA;
    dA.height  = M;
    dA.width   = K;
    dA.nnz     = A_values.size(0);
    dA.row_ptr = A_row_ptr.data_ptr<int>();
    dA.col_idx = A_col_idx.data_ptr<int>();
    dA.values  = A_values.data_ptr<float>();

    csrspmm::DenseMatrix dB;
    dB.height = B.size(0);
    dB.width  = B.size(1);
    dB.data   = B.data_ptr<float>();

    csrspmm::DenseMatrix dC;
    dC.height = C.size(0);
    dC.width  = C.size(1);
    dC.data   = C.data_ptr<float>();

    csrspmm::launch_warp_per_row(dA, dB, dC, alpha, beta);

    return C;
}
