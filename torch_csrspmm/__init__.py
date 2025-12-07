import torch
#  from .torch_csrspmm import *
from ._C import csrspmm_naive as _csrspmm_naive


def csrspmm_naive(A_rowptr, A_colidx, A_values, B, alpha, beta):
    return _csrspmm_naive(A_rowptr, A_colidx, A_values, B, alpha, beta)
