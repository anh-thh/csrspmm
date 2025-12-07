#include <torch/extension.h>

// Declare the function implemented in naive_wrapper.cpp
torch::Tensor naive_spmm_forward(
    torch::Tensor crow,
    torch::Tensor col,
    torch::Tensor values,
    torch::Tensor B
);

// Expose to Python via PyBind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "naive_spmm",
        &naive_spmm_forward,
        "Naive CSR × Dense SpMM (CUDA)"
    );
}
