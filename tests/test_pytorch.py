import torch
import random
import numpy as np

from csrspmm_torch import naive_spmm, naive_shared_spmm


def generate_random_csr(M, N, density=0.1):
    dense = torch.rand(M, N)
    dense = dense * (torch.rand(M, N) > density)

    crow = [0]
    col  = []
    val  = []

    nnz = 0
    for i in range(M):
        for j in range(N):
            if dense[i, j] != 0:
                col.append(j)
                val.append(float(dense[i, j]))
                nnz += 1
        crow.append(nnz)

    return (
        dense,
        torch.tensor(crow, dtype=torch.int32),
        torch.tensor(col, dtype=torch.int32),
        torch.tensor(val, dtype=torch.float32),
    )


def run_single_test(M, N, K, device):
    print(f"\n=== Running Test: {M} x {N} × {N} x {K} ===")

    A_dense, crow, col, val = generate_random_csr(M, N, density=0.7)
    B = torch.randn(N, K)

    crow = crow.to(device)
    col  = col.to(device)
    val  = val.to(device)
    B    = B.to(device)
    A_dense = A_dense.to(device)

    # ---- Run kernels ----
    C_naive = naive_spmm(crow, col, val, B)
    C_shared = naive_shared_spmm(crow, col, val, B)

    # ---- Reference ----
    C_ref = A_dense @ B

    # ---- Compare ----
    print("Checking naive...")
    print(torch.allclose(C_naive, C_ref, atol=1e-4, rtol=1e-4))

    print("Checking naive_shared...")
    print(torch.allclose(C_shared, C_ref, atol=1e-4, rtol=1e-4))

    print("Comparing naive vs naive_shared...")
    print(torch.allclose(C_naive, C_shared, atol=1e-4, rtol=1e-4))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    run_single_test(5, 5, 3, device)
    run_single_test(10, 12, 4, device)


if __name__ == "__main__":
    main()
