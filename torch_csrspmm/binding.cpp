#include <torch/extension.h>

// declare wrappers
torch::Tensor csrspmm_naive_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, float);
torch::Tensor csrspmm_naive_shared_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, float);

torch::Tensor csrspmm_warp_per_row_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, float);
torch::Tensor csrspmm_warp_per_row_fp4_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, float);

torch::Tensor csrspmm_warp_per_row_smem_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, float);
torch::Tensor csrspmm_warp_per_row_smem_fp4_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, float);


// pybind module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("csrspmm_naive", &csrspmm_naive_forward);
    m.def("csrspmm_naive_shared", &csrspmm_naive_shared_forward);

    m.def("csrspmm_warp_per_row", &csrspmm_warp_per_row_forward);
    m.def("csrspmm_warp_per_row_fp4", &csrspmm_warp_per_row_fp4_forward);

    m.def("csrspmm_warp_per_row_smem", &csrspmm_warp_per_row_smem_forward);
    m.def("csrspmm_warp_per_row_smem_fp4", &csrspmm_warp_per_row_smem_fp4_forward);
}
