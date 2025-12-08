#include <torch/extension.h>

// Declare the function implemented in naive_wrapper.cpp
torch::Tensor csrspmm_naive_forward(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, float
);

torch::Tensor csrspmm_warp_per_row_forward(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, float
);
// torch::Tensor csrspmm_warp_per_row_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);

torch::Tensor csrspmm_warp_per_row_fp4_forward(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, float
);

torch::Tensor csrspmm_warp_per_row_smem_forward(
    torch::Tensor, torch::Tensor, torch::Tensor, int, torch::Tensor, float, float
);



// Expose to Python via PyBind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("csrspmm_naive", &csrspmm_naive_forward, "CSR SpMM (naive)");
    m.def("csrspmm_warp_per_row", &csrspmm_warp_per_row_forward, "CSR SpMM (warp_per_row)");
    m.def("csrspmm_warp_per_row_smem", &csrspmm_warp_per_row_smem_forward, "CSR SpMM (warp_per_row_smem)");
    m.def("csrspmm_warp_per_row_fp4", &csrspmm_warp_per_row_fp4_forward, "CSR SpMM (warp_per_row_fp4)");
}
