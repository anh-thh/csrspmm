import torch
import csrspmm_torch


def naive_spmm(crow_indices: torch.Tensor,
               col_indices: torch.Tensor,
               values: torch.Tensor,
               dense_B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper around the naive CSR × dense SpMM CUDA kernel.

    Parameters
    ----------
    crow_indices : int32 CUDA tensor
        CSR row pointer array of shape (M+1,)
    col_indices : int32 CUDA tensor
        CSR column indices array of shape (nnz,)
    values : float32 CUDA tensor
        CSR non-zero values array of shape (nnz,)
    dense_B : float32 CUDA tensor
        Dense matrix B of shape (K, N)

    Returns
    -------
    torch.Tensor
        Dense matrix C = A * B of shape (M, N)
    """
    return csrspmm_torch.naive_spmm(
        crow_indices,
        col_indices,
        values,
        dense_B,
    )


def naive_spmm_shared(crow_indices: torch.Tensor,
                      col_indices: torch.Tensor,
                      values: torch.Tensor,
                      dense_B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper around the shared-memory CSR × dense SpMM CUDA kernel.

    Parameters
    ----------
    crow_indices : int32 CUDA tensor
        CSR row pointer array of shape (M+1,)
    col_indices : int32 CUDA tensor
        CSR column indices array of shape (nnz,)
    values : float32 CUDA tensor
        CSR non-zero values array of shape (nnz,)
    dense_B : float32 CUDA tensor
        Dense matrix B of shape (K, N)

    Returns
    -------
    torch.Tensor
        Dense matrix C = A * B of shape (M, N)
    """
    return csrspmm_torch.naive_spmm_shared(
        crow_indices,
        col_indices,
        values,
        dense_B,
    )
