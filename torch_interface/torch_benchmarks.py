import torch
import time
import numpy as np
import csrspmm_torch   # your custom CUDA extension


print(f"PyTorch version: {torch.__version__}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This benchmark requires a GPU.")
else:
    torch.cuda.init()
    DEVICE = torch.device("cuda")
    print("Running benchmark on GPU:", torch.cuda.get_device_name(0))


# ----------------------------------------------------------------------
# Helper: initialize random sparse CSR + dense B
# ----------------------------------------------------------------------
def initialize_matrices(M, N, K, sparsity=0.7):
    A = torch.rand(M, K, device=DEVICE, dtype=torch.float32)
    mask = torch.rand_like(A) > sparsity
    A = A * mask

    B = torch.rand(K, N, device=DEVICE, dtype=torch.float32)
    C = torch.zeros(M, N, device=DEVICE, dtype=torch.float32)

    A_csr = A.to_sparse_csr()

    return A, B, C, A_csr


# ----------------------------------------------------------------------
# Generic timer
# ----------------------------------------------------------------------
def run_benchmark(fn, *args, n_warmup=10, iterations=100, **kwargs):
    # warmup
    for _ in range(n_warmup):
        fn(*args, **kwargs)
        torch.cuda.synchronize()

    latencies = []
    for _ in range(iterations):
        t0 = time.time()
        fn(*args, **kwargs)
        torch.cuda.synchronize()
        latencies.append(time.time() - t0)

    return latencies


# ----------------------------------------------------------------------
# Baseline PyTorch ops
# ----------------------------------------------------------------------
def fn_torch_sparse_mm(alpha, A_csr, beta, B, C):
    return alpha * torch.sparse.mm(A_csr, B) + beta * C


def fn_torch_sparse_addmm(alpha, A_csr, beta, B, C):
    return torch.sparse.addmm(C, A_csr, B, beta=beta, alpha=alpha)


# ----------------------------------------------------------------------
# Our custom CUDA kernel (naive)
# ----------------------------------------------------------------------
def fn_custom_naive(alpha, A_csr, beta, B, C):
    # Convert CSR indices to int32 for CUDA
    crow = A_csr.crow_indices().to(torch.int32)
    col  = A_csr.col_indices().to(torch.int32)
    val  = A_csr.values()  # already float32

    # call CUDA kernel
    C_tmp = csrspmm_torch.naive_spmm(crow, col, val, B)

    # apply alpha * A*B + beta*C
    return alpha * C_tmp + beta * C


# ----------------------------------------------------------------------
# Reporting helper
# ----------------------------------------------------------------------
def report(name, times):
    avg = np.mean(times) * 1000
    std = np.std(times) * 1000
    print(f"{name:<25}: {avg:8.3f} ms  +/- {std:6.3f}")


# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":

    M, N, K = 1024, 1024, 1024
    alpha = 1.0
    beta = 0.5
    sparsity = 0.7

    n_warmup = 10
    iterations = 200

    print("\n---- Running Benchmarks ----")

    A, B, C, A_csr = initialize_matrices(M, N, K, sparsity)

    # Torch baselines
    t_mm = run_benchmark(fn_torch_sparse_mm,
                         alpha, A_csr, beta, B, C,
                         n_warmup=n_warmup, iterations=iterations)

    t_addmm = run_benchmark(fn_torch_sparse_addmm,
                            alpha, A_csr, beta, B, C,
                            n_warmup=n_warmup, iterations=iterations)

    # Custom CUDA kernel
    t_custom = run_benchmark(fn_custom_naive,
                             alpha, A_csr, beta, B, C,
                             n_warmup=n_warmup, iterations=iterations)

    print("\n---- Results ----")
    report("torch.sparse.mm", t_mm)
    report("torch.sparse.addmm", t_addmm)
    report("custom_naive_spmm", t_custom)

    print("\nBenchmark completed.")
