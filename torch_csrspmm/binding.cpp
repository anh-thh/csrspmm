#include <torch/extension.h>

// Declare the function implemented in naive_wrapper.cpp
torch::Tensor csrspmm_naive_forward( torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
// torch::Tensor csrspmm_warp_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);

// Expose to Python via PyBind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("csrspmm_naive", &csrspmm_naive_forward, "CSR SpMM (naive)");
    // m.def("csrspmm_warp", &csrspmm_warp_forward, "CSR SpMM (warp_per_row)");
}
