import torch
import csrspmm_torch

def naive_spmm(crow_indices, col_indices, values, dense_B):
    return csrspmm_torch.naive_spmm(
        crow_indices,
        col_indices,
        values,
        dense_B
    )
