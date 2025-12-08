#include <torch/extension.h>

// declare wrappers
torch::Tensor csrspmm_naive_forward(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, float);

torch::Tensor csrspmm_naive_shared_forward(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, float);

torch::Tensor csrspmm_warp_per_row_forward(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, float);

torch::Tensor csrspmm_warp_per_row_fp4_forward(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, float);

torch::Tensor csrspmm_warp_per_row_smem_forward(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, float);

torch::Tensor csrspmm_warp_per_row_smem_fp4_forward(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, float);


// pybind module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("csrspmm_naive",        &csrspmm_naive_forward,        "CSR SpMM (naive)");
    m.def("csrspmm_naive_shared", &csrspmm_naive_shared_forward, "CSR SpMM (naive_shared)");

    m.def("csrspmm_warp_per_row",     &csrspmm_warp_per_row_forward,     "CSR SpMM (warp_per_row)");
    m.def("csrspmm_warp_per_row_fp4", &csrspmm_warp_per_row_fp4_forward, "CSR SpMM (warp_per_row_fp4)");

    m.def("csrspmm_warp_per_row_smem",
          &csrspmm_warp_per_row_smem_forward,
          "CSR SpMM (warp_per_row_smem)");

    m.def("csrspmm_warp_per_row_smem_fp4",
          &csrspmm_warp_per_row_smem_fp4_forward,
          "CSR SpMM (warp_per_row_smem_fp4)");

    // m.def("csrspmm_warp_per_row_smem_forward",
    //   &csrspmm_warp_per_row_smem_forward,
    //   "CSRSpMM WarpPerRow SMEM forward kernel");

    // m.def("csrspmm_warp_per_row_smem_fp4_forward",
    //   &csrspmm_warp_per_row_smem_fp4_forward,
    //   "CSRSpMM WarpPerRow SMEM FP4 forward kernel");
}
