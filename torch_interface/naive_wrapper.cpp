// torch_interface/naive_wrapper.cpp
#include <torch/extension.h>
#include <iostream>

#include "csrspmm/matrix.h"
#include "csrspmm/csr_utils.h"
#include "csrspmm/dense_utils.h"
#include "csrspmm/spmm_launcher.cuh"

using torch::Tensor;

torch::Tensor csrspmm_naive_forward(
    Tensor crow,      // int32 row_ptr
    Tensor col,       // int32 col_idx
    Tensor values,    // float32 values
    Tensor B          // float32 dense input
) {
    TORCH_CHECK(crow.is_cuda(),   "crow must be a CUDA tensor");
    TORCH_CHECK(col.is_cuda(),    "col must be a CUDA tensor");
    TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(),      "B must be a CUDA tensor");

    TORCH_CHECK(crow.dtype() == torch::kInt32,  "crow must be int32");
    TORCH_CHECK(col.dtype()  == torch::kInt32,  "col must be int32");
    TORCH_CHECK(values.dtype() == torch::kFloat32, "values must be float32");

    int M = crow.size(0) - 1;
    int K = B.size(0);  // consistent with your existing code
    int N = B.size(1);

    Tensor C = torch::zeros({M, N}, B.options());

    csrspmm::CSRMatrix dA;
    dA.height  = M;
    dA.width   = K;
    dA.nnz     = values.size(0);
    dA.row_ptr = crow.data_ptr<int>();
    dA.col_idx = col.data_ptr<int>();
    dA.values  = values.data_ptr<float>();

    csrspmm::DenseMatrix dB;
    dB.height = B.size(0);
    dB.width  = B.size(1);
    dB.data   = B.data_ptr<float>();

    csrspmm::DenseMatrix dC;
    dC.height = C.size(0);
    dC.width  = C.size(1);
    dC.data   = C.data_ptr<float>();

    // alpha = 1.0, beta = 0.0 (you can parameterize later if needed)
    csrspmm::launch_naive(dA, dB, dC, 1.0f, 0.0f);

    return C;
}
