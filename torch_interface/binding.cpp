#include <torch/extension.h>

torch::Tensor naive_spmm_forward(
    torch::Tensor crow,
    torch::Tensor col,
    torch::Tensor values,
    torch::Tensor B);

torch::Tensor naive_spmm_shared_forward(
    torch::Tensor crow,
    torch::Tensor col,
    torch::Tensor values,
    torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_spmm", &naive_spmm_forward,
          "Naive CSR × Dense SpMM");

    m.def("naive_spmm_shared", &naive_spmm_shared_forward,
          "Shared-memory CSR × Dense SpMM");
}
