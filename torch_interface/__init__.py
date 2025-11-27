# torch_interface/__init__.py
# Public Python API for the CSR SpMM CUDA extension

import torch
import csrspmm_torch


# ============================================================
#  Base kernels
# ============================================================

def csrspmm_naive(crow, col, values, B):
    """
    Naive CSR × Dense SpMM
    """
    return csrspmm_torch.csrspmm_naive(crow, col, values, B)


# ============================================================
#  Warp-per-row kernels
# ============================================================

def csrspmm_warp(crow, col, values, B):
    """
    Warp-per-row CSR SpMM
    """
    return csrspmm_torch.csrspmm_warp(crow, col, values, B)


def csrspmm_warp_smem(crow, col, values, B):
    """
    Warp-per-row with shared memory
    """
    return csrspmm_torch.csrspmm_warp_smem(crow, col, values, B)


# ============================================================
#  FP4 (float4) vectorized kernels
# ============================================================

def csrspmm_warp_fp4(crow, col, values, B):
    """
    Warp-per-row, float4 vectorized
    """
    return csrspmm_torch.csrspmm_warp_fp4(crow, col, values, B)


def csrspmm_warp_smem_fp4(crow, col, values, B):
    """
    Warp-per-row + shared memory + float4 vectorized
    """
    return csrspmm_torch.csrspmm_warp_smem_fp4(crow, col, values, B)
