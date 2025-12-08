import torch
#  from .torch_csrspmm import *
from ._C import csrspmm_naive as _csrspmm_naive
from ._C import csrspmm_warp_per_row as _csrspmm_warp_per_row
from ._C import csrspmm_warp_per_row_smem as _csrspmm_warp_per_row_smem
from ._C import csrspmm_warp_per_row_fp4 as _csrspmm_warp_per_row_fp4


def csrspmm_naive(A_rowptr, A_colidx, A_values, B, alpha, beta):
    return _csrspmm_naive(A_rowptr, A_colidx, A_values, B, alpha, beta)

def csrspmm_warp_per_row(A_rowptr, A_colidx, A_values, B, alpha, beta):
    return _csrspmm_warp_per_row(A_rowptr, A_colidx, A_values, B, alpha, beta)

def csrspmm_warp_per_row_smem(A_rowptr, A_colidx, A_values, A_max_row_nnz, B, alpha, beta):
    return _csrspmm_warp_per_row_smem(A_rowptr, A_colidx, A_values, A_max_row_nnz, B, alpha, beta)

def csrspmm_warp_per_row_fp4(A_rowptr, A_colidx, A_values, B, alpha, beta):
    return _csrspmm_warp_per_row_fp4(A_rowptr, A_colidx, A_values, B, alpha, beta)
