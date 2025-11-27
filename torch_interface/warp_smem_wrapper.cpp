// torch_interface/warp_smem_wrapper.cpp
#include <torch/extension.h>
#include <cstdint>

#include "csrspmm/matrix.h"
#include "csrspmm/csr_utils.h"
#include "csrspmm/dense_utils.h"
#include "csrspmm/spmm_launcher.cuh"

using torch::Tensor;

static int compute_max_row_nnz(const Tensor& crow_cpu) {
    auto crow_acc = crow_cpu.data_ptr<int>();
    int M = static_cast<int>(crow_cpu.size(0)) - 1;
    int max_nnz = 0;
    for (int i = 0; i < M; ++i) {
        int nnz = crow_acc[i + 1] - crow_acc[i];
        if (nnz > max_nnz) max_nnz = nnz;
    }
    return max_nnz;
}

torch::Tensor csrspmm_warp_smem_forward(
    Tensor crow,
    Tensor col,
    Tensor values,
    Tensor B
) {
    // ---- Basic checks ----
    TORCH_CHECK(crow.is_cuda(),   "crow must be a CUDA tensor");
    TORCH_CHECK(col.is_cuda(),    "col must be a CUDA tensor");
    TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(),      "B must be a CUDA tensor");

    TORCH_CHECK(crow.dtype()   == torch::kInt32,   "crow must be int32");
    TORCH_CHECK(col.dtype()    == torch::kInt32,   "col must be int32");
    TORCH_CHECK(values.dtype() == torch::kFloat32, "values must be float32");

    int M = static_cast<int>(crow.size(0)) - 1;
    int K = static_cast<int>(B.size(0));  // rows of B
    int N = static_cast<int>(B.size(1));  // cols of B

    // ---- Compute max nnz/row on CPU ----
    Tensor crow_cpu = crow.to(torch::kCPU);
    int A_max_row_nnz = compute_max_row_nnz(crow_cpu);

    // ---- Allocate output ----
    Tensor C = torch::zeros({M, N}, B.options());

    // ---- Build CSRMatrix view ----
    csrspmm::CSRMatrix dA;
    dA.height      = M;
    dA.width       = K;
    dA.nnz         = static_cast<int>(values.size(0));
    dA.row_ptr     = crow.data_ptr<int>();
    dA.col_idx     = col.data_ptr<int>();
    dA.values      = values.data_ptr<float>();
    dA.max_row_nnz = A_max_row_nnz;     // ✅ IMPORTANT: used by launcher & kernel

    // ---- Build DenseMatrix views ----
    csrspmm::DenseMatrix dB;
    dB.height = K;
    dB.width  = N;
    dB.data   = B.data_ptr<float>();

    csrspmm::DenseMatrix dC;
    dC.height = M;
    dC.width  = N;
    dC.data   = C.data_ptr<float>();

    // ---- Launch shared-memory warp kernel ----
    csrspmm::launch_warp_per_row_smem(dA, dB, dC, 1.0f, 0.0f);

    return C;
}
