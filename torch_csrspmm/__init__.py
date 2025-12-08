import torch  # not strictly needed, but fine to keep

# Import compiled extension symbols
from ._C import (
    csrspmm_naive as _csrspmm_naive,
    csrspmm_naive_shared as _csrspmm_naive_shared,
    csrspmm_warp_per_row as _csrspmm_warp_per_row,
    csrspmm_warp_per_row_fp4 as _csrspmm_warp_per_row_fp4,
    csrspmm_warp_per_row_smem as _csrspmm_warp_per_row_smem,
    csrspmm_warp_per_row_smem_fp4 as _csrspmm_warp_per_row_smem_fp4,
)

# Thin Python wrappers (just for a stable public API)
def csrspmm_naive(A_rowptr, A_colidx, A_values, B, alpha, beta):
    return _csrspmm_naive(A_rowptr, A_colidx, A_values, B, alpha, beta)

def csrspmm_naive_shared(A_rowptr, A_colidx, A_values, B, alpha, beta):
    return _csrspmm_naive_shared(A_rowptr, A_colidx, A_values, B, alpha, beta)

def csrspmm_warp_per_row(A_rowptr, A_colidx, A_values, B, alpha, beta):
    return _csrspmm_warp_per_row(A_rowptr, A_colidx, A_values, B, alpha, beta)

def csrspmm_warp_per_row_fp4(A_rowptr, A_colidx, A_values, B, alpha, beta):
    return _csrspmm_warp_per_row_fp4(A_rowptr, A_colidx, A_values, B, alpha, beta)

def csrspmm_warp_per_row_smem(A_rowptr, A_colidx, A_values, B, alpha, beta):
    return _csrspmm_warp_per_row_smem(A_rowptr, A_colidx, A_values, B, alpha, beta)

def csrspmm_warp_per_row_smem_fp4(A_rowptr, A_colidx, A_values, B, alpha, beta):
    return _csrspmm_warp_per_row_smem_fp4(A_rowptr, A_colidx, A_values, B, alpha, beta)
