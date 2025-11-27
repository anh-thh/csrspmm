import torch
import time
import numpy as np
import csrspmm_torch   # compiled CUDA extension
import argparse


print(f"PyTorch version: {torch.__version__}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This benchmark requires a GPU.")
else:
    torch.cuda.init()
    DEVICE = torch.device("cuda")
    print("Running benchmark on GPU:", torch.cuda.get_device_name(0))


# ========================================================================
#  Helper: Initialize random sparse CSR + dense B
# ========================================================================
def initialize_matrices(M, N, K, sparsity=0.7):
    A = torch.rand(M, K, device=DEVICE, dtype=torch.float32)
    mask = torch.rand_like(A) > sparsity
    A = A * mask

    B = torch.rand(K, N, device=DEVICE, dtype=torch.float32)
    C = torch.zeros(M, N, device=DEVICE, dtype=torch.float32)

    A_csr = A.to_sparse_csr()

    return A, B, C, A_csr


# ========================================================================
#  Timing helpers
# ========================================================================
def run_benchmark(fn, *args, n_warmup=10, iterations=100, **kwargs):
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


def report(name, times):
    avg = np.mean(times) * 1000  # ms
    std = np.std(times) * 1000
    print(f"{name:<28}: {avg:8.3f} ms  +/- {std:6.3f}")


# ========================================================================
#  Baseline PyTorch SpMM
# ========================================================================
def fn_torch_spmm(alpha, A_csr, beta, B, C):
    return alpha * torch.sparse.mm(A_csr, B) + beta * C


def fn_torch_addmm(alpha, A_csr, beta, B, C):
    return torch.sparse.addmm(C, A_csr, B, beta=beta, alpha=alpha)


# ========================================================================
#  Our custom kernels
# ========================================================================
def extract_csr(A_csr):
    crow = A_csr.crow_indices().to(torch.int32)
    col  = A_csr.col_indices().to(torch.int32)
    val  = A_csr.values()  # float32
    return crow, col, val


def fn_custom_naive(alpha, A_csr, beta, B, C):
    crow, col, val = extract_csr(A_csr)
    C_tmp = csrspmm_torch.csrspmm_naive(crow, col, val, B)
    return alpha * C_tmp + beta * C


def fn_custom_warp(alpha, A_csr, beta, B, C):
    crow, col, val = extract_csr(A_csr)
    C_tmp = csrspmm_torch.csrspmm_warp(crow, col, val, B)
    return alpha * C_tmp + beta * C


def fn_custom_warp_smem(alpha, A_csr, beta, B, C):
    crow, col, val = extract_csr(A_csr)
    C_tmp = csrspmm_torch.csrspmm_warp_smem(crow, col, val, B)
    return alpha * C_tmp + beta * C


def fn_custom_warp_fp4(alpha, A_csr, beta, B, C):
    crow, col, val = extract_csr(A_csr)
    C_tmp = csrspmm_torch.csrspmm_warp_fp4(crow, col, val, B)
    return alpha * C_tmp + beta * C


def fn_custom_warp_smem_fp4(alpha, A_csr, beta, B, C):
    crow, col, val = extract_csr(A_csr)
    C_tmp = csrspmm_torch.csrspmm_warp_smem_fp4(crow, col, val, B)
    return alpha * C_tmp + beta * C

# ============================================================
# CLI argument parser for selective kernel benchmarking
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--kernel",
                    type=str,
                    default="all",
                    choices=["all", "naive", "warp", "warp_smem", "warp_fp4", "warp_smem_fp4"],
                    help="Select a single kernel to benchmark")
args = parser.parse_args()


# ========================================================================
#  MAIN Benchmark
# ========================================================================
if __name__ == "__main__":

    M, N, K = 1024, 1024, 1024
    alpha = 1.0
    beta  = 0.5
    sparsity = 0.7

    n_warmup = 10
    iterations = 200

    print("\n---- Initializing matrices ----")
    A, B, C, A_csr = initialize_matrices(M, N, K, sparsity)

    print("\n---- Running Benchmarks ----")

    def bench(name, fn):
        times = run_benchmark(fn, alpha, A_csr, beta, B, C,
                            n_warmup=n_warmup, iterations=iterations)
        report(name, times)


    if args.kernel == "all" or args.kernel == "naive":
        bench("custom_csrspmm_naive", fn_custom_naive)

    if args.kernel == "all" or args.kernel == "warp":
        bench("custom_csrspmm_warp", fn_custom_warp)

    if args.kernel == "all" or args.kernel == "warp_smem":
        bench("custom_csrspmm_warp_smem", fn_custom_warp_smem)

    if args.kernel == "all" or args.kernel == "warp_fp4":
        bench("custom_csrspmm_warp_fp4", fn_custom_warp_fp4)

    if args.kernel == "all" or args.kernel == "warp_smem_fp4":
        bench("custom_csrspmm_warp_smem_fp4", fn_custom_warp_smem_fp4)

    # Only run PyTorch baselines in full benchmark mode
    if args.kernel == "all":
        bench("torch.sparse.mm", fn_torch_spmm)
        bench("torch.sparse.addmm", fn_torch_addmm)

    # ====================================================================
    # Print results
    # ====================================================================
    print("\n---- Results ----")
    report("torch.sparse.mm",            t_spmm)
    report("torch.sparse.addmm",         t_addmm)
    report("custom_csrspmm_naive",       t_naive)
    report("custom_csrspmm_warp",        t_warp)
    report("custom_csrspmm_warp_smem",   t_warp_smem)
    report("custom_csrspmm_warp_fp4",    t_warp_fp4)
    report("custom_csrspmm_warp_smem_fp4", t_warp_smem_fp4)

    print("\nBenchmark completed.")
