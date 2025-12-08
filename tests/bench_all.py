import torch
import time
import numpy as np
import torch_csrspmm  # your custom CUDA kernels


print(f"PyTorch version: {torch.__version__}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available! Benchmark requires GPU.")
else:
    DEVICE = torch.device("cuda")
    print("Running benchmark on GPU:", torch.cuda.get_device_name(0))


# ---------------------------------------------------------
# Matrix Initialization
# ---------------------------------------------------------
def initialize_matrices(M, N, K, sparsity=0.70):
    A = torch.rand(M, K, device=DEVICE, dtype=torch.float32)
    mask = torch.rand_like(A) > sparsity
    A = A * mask

    B = torch.rand(K, N, device=DEVICE, dtype=torch.float32)
    C = torch.zeros(M, N, device=DEVICE)

    A_csr = A.to_sparse_csr()

    return A, B, C, A_csr


# ---------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------
def run_benchmark(fn, *args, n_warmup=10, iterations=100, **kwargs):
    # Warmup (avoid cold-start + kernels being JIT’d)
    for _ in range(n_warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    latencies = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        t0 = time.time()
        fn(*args, **kwargs)
        torch.cuda.synchronize()
        latencies.append(time.time() - t0)

    return np.array(latencies)


# ---------------------------------------------------------
# Kernel wrappers
# ---------------------------------------------------------
def extract_csr(A_csr):
    row = A_csr.crow_indices().to(torch.int32)
    col = A_csr.col_indices().to(torch.int32)
    val = A_csr.values()
    return row, col, val


def fn_torch_sparse_mm(alpha, A_csr, beta, B, C):
    return alpha * torch.sparse.mm(A_csr, B) + beta * C


def fn_torch_sparse_addmm(alpha, A_csr, beta, B, C):
    return torch.sparse.addmm(C, A_csr, B, beta=beta, alpha=alpha)


def fn_custom(alpha, A_csr, beta, B, kernel):
    A_row_ptr, A_col_idx, A_values = extract_csr(A_csr)
    return kernel(A_row_ptr, A_col_idx, A_values, B, alpha, beta)


# ---------------------------------------------------------
# Reporting utilities
# ---------------------------------------------------------
def report(name, times, M, N, K, nnz):
    avg = times.mean() * 1000
    std = times.std() * 1000

    # GFLOPS for SpMM = 2 * nnz * N
    gflops = (2 * nnz * N) / (avg / 1000) / 1e9

    print(f"{name:<30}: {avg:8.3f} ms  +/- {std:6.3f}   ({gflops:6.2f} GFLOP/s)")


# ---------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------
if __name__ == "__main__":

    M, N, K = 2048, 2048, 2048
    sparsity = 0.70

    alpha = 1.0
    beta = 0.0

    iterations = 50
    warmup = 5

    print(f"\nBenchmarking SPMM: M={M}, N={N}, K={K}, sparsity={sparsity}")

    A, B, C, A_csr = initialize_matrices(M, N, K, sparsity)
    nnz = A_csr.values().numel()

    # ------------------------
    # PyTorch baselines
    # ------------------------
    t_mm = run_benchmark(
        fn_torch_sparse_mm, alpha, A_csr, beta, B, C,
        n_warmup=warmup, iterations=iterations
    )
    t_addmm = run_benchmark(
        fn_torch_sparse_addmm, alpha, A_csr, beta, B, C,
        n_warmup=warmup, iterations=iterations
    )

    # ------------------------
    # Custom kernels
    # ------------------------
    kernels = {
        "naive": torch_csrspmm.csrspmm_naive,
        "naive_shared": torch_csrspmm.csrspmm_naive_shared,
        "warp_per_row": torch_csrspmm.csrspmm_warp_per_row,
        "warp_per_row_fp4": torch_csrspmm.csrspmm_warp_per_row_fp4,
        "warp_per_row_smem": torch_csrspmm.csrspmm_warp_per_row_smem,
        "warp_per_row_smem_fp4": torch_csrspmm.csrspmm_warp_per_row_smem_fp4,
    }

    timings = {}

    for name, kernel in kernels.items():
        print(f"Running kernel: {name}")
        t = run_benchmark(
            fn_custom, alpha, A_csr, beta, B,
            kernel=kernel,
            n_warmup=warmup,
            iterations=iterations
        )
        timings[name] = t

        # Correctness check
        ref = fn_torch_sparse_mm(alpha, A_csr, beta, B, C)
        out = fn_custom(alpha, A_csr, beta, B, kernel)
        max_err = (out - ref).abs().max().item()

        print(f"  Max error vs PyTorch: {max_err:e}")

    # ------------------------
    # Report results
    # ------------------------
    print("\n======== RESULTS =========\n")
    report("torch.sparse.mm", t_mm, M, N, K, nnz)
    report("torch.sparse.addmm", t_addmm, M, N, K, nnz)

    for name in kernels.keys():
        report(name, timings[name], M, N, K, nnz)

    print("\nBenchmark complete.\n")
