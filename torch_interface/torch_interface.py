# pytorch interface for sparse matrix-matrix multiplication (SpMM)
# This wraps the custom CUDA extension built from spmm_ops.cpp + spmm_kernel_launcher.cu

import torch
import sparse_cuda_ext  # built by setup.py (a .so/.pyd extension)


def csr_spmm(alpha: float,
             A_csr: torch.Tensor,
             beta: float,
             B: torch.Tensor,
             C: torch.Tensor) -> torch.Tensor:
    """
    PyTorch-facing wrapper.

    Parameters:
        alpha, beta: scalars
        A_csr:  sparse CSR tensor of shape (M, K_in)
        B:      dense tensor of shape (K_in, N_out)
        C:      dense tensor of shape (M, N_out)

    Returns:
        Dense tensor of shape (M, N_out) with:
            alpha * A_csr @ B + beta * C
    """
    if not A_csr.is_sparse_csr:
        raise TypeError("A_csr must be a sparse CSR tensor")

    if not (A_csr.device.type == "cuda"
            and B.device.type == "cuda"
            and C.device.type == "cuda"):
        raise RuntimeError("All tensors must be on CUDA device")

    return sparse_cuda_ext.csr_spmm(float(alpha), A_csr, float(beta), B, C)
