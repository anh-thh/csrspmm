# pytorch interface for sparse matrix-matrix multiplication (SpMM)
# see https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
import torch 
from setuptools import setup, Extension
from torch.utils import cpp_extension

def csr_spmm(alph, A_csr, beta, B, C):
    raise NotImplementedError
