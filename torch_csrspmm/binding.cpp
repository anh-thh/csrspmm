#include <torch/extension.h>

// Declare the function implemented in naive_wrapper.cpp
torch::Tensor csrspmm_naive_forward(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, float
);

torch::Tensor csrspmm_warp_per_row_forward(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, float
);

// Expose to Python via PyBind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("csrspmm_naive", &csrspmm_naive_forward, "CSR SpMM (naive)");
    m.def("csrspmm_warp_per_row", &csrspmm_warp_per_row_forward, "CSR SpMM (warp_per_row)");
}
