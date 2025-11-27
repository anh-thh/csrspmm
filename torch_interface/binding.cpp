#include <torch/extension.h>

torch::Tensor csrspmm_naive_forward(      torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
torch::Tensor csrspmm_warp_forward(       torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
torch::Tensor csrspmm_warp_smem_forward(  torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
torch::Tensor csrspmm_warp_smem_fp4_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
// 🔹 NEW:
torch::Tensor csrspmm_warp_fp4_forward(   torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("csrspmm_naive",       &csrspmm_naive_forward,       "CSR SpMM (naive)");
    m.def("csrspmm_warp",        &csrspmm_warp_forward,        "CSR SpMM (warp_per_row)");
    m.def("csrspmm_warp_smem",   &csrspmm_warp_smem_forward,   "CSR SpMM (warp_per_row + smem)");
    m.def("csrspmm_warp_smem_fp4",
           &csrspmm_warp_smem_fp4_forward,
           "CSR SpMM (warp_per_row + smem + float4)");
    // 🔹 NEW:
    m.def("csrspmm_warp_fp4",
           &csrspmm_warp_fp4_forward,
           "CSR SpMM (warp_per_row float4, no shared mem)");
}
