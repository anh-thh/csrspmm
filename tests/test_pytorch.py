import torch
import random
import numpy as np
import torch_csrspmm


def generate_random_csr(M, N, density=0.7):
    dense = torch.rand(M, N)
    mask = (torch.rand(M, N) > density)
    dense = dense * mask

    crow = [0]
    col = []
    val = []
    nnz = 0
    for i in range(M):
        for j in range(N):
            if dense[i, j] != 0:
                col.append(j)
                val.append(float(dense[i, j]))
                nnz += 1
        crow.append(nnz)

    return dense, torch.tensor(crow, dtype=torch.int32), \
           torch.tensor(col, dtype=torch.int32), \
           torch.tensor(val, dtype=torch.float32)


def run_single_test(M, N, K, device, algo):
    print(f"\n=== Running Test [{algo}]   ({M}x{N}) * ({N}x{K}) ===")

    A_dense, crow, col, val = generate_random_csr(M, N)
    B = torch.randn(N, K, dtype=torch.float32)

    crow = crow.to(device)
    col  = col.to(device)
    val  = val.to(device)
    B    = B.to(device)
    A_dense = A_dense.to(device)

    alpha = 1.0
    beta  = 1.0

    # Dispatch kernel
    fn = {
        "naive":               torch_csrspmm.csrspmm_naive,
        "naive_shared":        torch_csrspmm.csrspmm_naive_shared,
        "warp_per_row":        torch_csrspmm.csrspmm_warp_per_row,
        "warp_per_row_fp4":    torch_csrspmm.csrspmm_warp_per_row_fp4,
        "warp_per_row_smem":   torch_csrspmm.csrspmm_warp_per_row_smem,
        "warp_per_row_smem_fp4": torch_csrspmm.csrspmm_warp_per_row_smem_fp4,
    }[algo]

    C1 = fn(crow, col, val, B, alpha, beta)
    C2 = A_dense @ B

    if torch.allclose(C1, C2, atol=1e-4, rtol=1e-4):
        print(f"[{algo}] PASS")
    else:
        print(f"[{algo}] FAIL")
        print("Max error:", (C1 - C2).abs().max().item())


def main():
    device = "cuda"
    algos = [
        "naive",
        "naive_shared",
        "warp_per_row",
        "warp_per_row_fp4",
        "warp_per_row_smem",
        "warp_per_row_smem_fp4",
    ]

    for algo in algos:
        run_single_test(4, 4, 8, device, algo)
        run_single_test(8, 8, 16, device, algo)


if __name__ == "__main__":
    main()
