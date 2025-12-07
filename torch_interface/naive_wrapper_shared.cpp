#include <torch/extension.h>
#include "csrspmm/matrix.h"
#include "csrspmm/spmm_launcher.cuh"

using torch::Tensor;

Tensor naive_spmm_shared_forward(
    Tensor crow,
    Tensor col,
    Tensor values,
    Tensor B
) {
    TORCH_CHECK(crow.is_cuda(), "crow must be CUDA");
    TORCH_CHECK(col.is_cuda(), "col must be CUDA");
    TORCH_CHECK(values.is_cuda(), "values must be CUDA");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA");

    int M = crow.size(0) - 1;
    int K = B.size(0);
    int N = B.size(1);

    Tensor C = torch::zeros({M, N}, B.options());

    csrspmm::CSRMatrix A;
    A.height  = M;
    A.width   = K;
    A.nnz     = values.numel();
    A.row_ptr = crow.data_ptr<int>();
    A.col_idx = col.data_ptr<int>();
    A.values  = values.data_ptr<float>();

    csrspmm::DenseMatrix dB;
    dB.height = K;
    dB.width  = N;
    dB.data   = B.data_ptr<float>();

    csrspmm::DenseMatrix dC;
    dC.height = M;
    dC.width  = N;
    dC.data   = C.data_ptr<float>();

    launch_naive_shared(A, dB, dC, 1.0f, 0.0f);

    return C;
}
