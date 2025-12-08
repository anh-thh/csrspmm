import torch
import random
import numpy as np

# Import your interface
import torch_csrspmm


def generate_random_csr(M, N, density=0.1):
    """
    Generate a random CSR matrix with given density.
    """

    # Dense matrix
    dense = torch.rand(M, N)

    mask = (torch.rand(M, N) > density)
    dense = dense * mask  # zero out many entries

    # Convert to CSR
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

    crow = torch.tensor(crow, dtype=torch.int32)
    col  = torch.tensor(col, dtype=torch.int32)
    val  = torch.tensor(val, dtype=torch.float32)

    return dense, crow, col, val


def run_single_test(M, N, K, device, algo="naive"):
    print(f"\n=== Running Test: [{M} x {N}] x [{N} x {K}] ===")

    # Generate random CSR matrix
    A_dense, crow, col, val = generate_random_csr(M, N, density=0.7)

    # Dense B
    B = torch.randn(N, K, dtype=torch.float32)

    # Move to device (CPU or GPU)
    crow = crow.to(device)
    col  = col.to(device)
    val  = val.to(device)
    B    = B.to(device)
    A_dense = A_dense.to(device)

    # ---- Run your kernel ----
    alpha, beta = 1, 1 
    if algo == "naive":
        C1 = torch_csrspmm.csrspmm_naive(crow, col, val, B, alpha, beta)
    elif algo == "naive_shared":
        C1 = torch_csrspmm.csrspmm_naive_shared(crow, col, val, B, alpha, beta)
    elif algo == "warp_per_row": 
        C1 = torch_csrspmm.csrspmm_warp_per_row(crow, col, val, B, alpha, beta)
    elif algo == "warp_per_row_fp4": 
        C1 = torch_csrspmm.csrspmm_warp_per_row_fp4(crow, col, val, B, alpha, beta)
    else: 
        raise NotImplementedError("Algorithm currently not supported")

    # ---- Reference result ----
    C2 = A_dense @ B

    # ---- Compare ----
    if torch.allclose(C1, C2, atol=1e-4, rtol=1e-4):
        print(f"[{algo}] PASS — Output matches PyTorch matmul")
    else:
        print(f"[{algo}] FAIL — Mismatch detected")
        print("Your output C1:\n", C1)
        print("Reference C2:\n", C2)
        diff = (C1 - C2).abs().max()
        print("Max error:", diff.item())


def main():
    # Pick device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    
    algos = ["naive", "naive_shared", "warp_per_row", "warp_per_row_fp4"]
    for algo in algos:
        # ---- Small tests ----
        run_single_test(4, 4, 8, device, algo)
        run_single_test(8, 8, 16, device, algo)

        # ---- Medium tests ----
        for _ in range(5):
            M = random.randint(2, 10)
            N = random.randint(2, 10)
            K = random.randint(2, 10)
            run_single_test(2**M, 2**N, 2**K, device, algo)


if __name__ == "__main__":
    main()
