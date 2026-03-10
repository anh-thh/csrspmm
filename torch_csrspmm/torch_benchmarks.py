import torch
import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch_csrspmm

DEVICE = torch.device("cuda")

# -------------------------------------------------------------
# CONFIGS
# -------------------------------------------------------------
CONFIGS = [
    ("small", 512, 1024, 64),
    ("medium", 1024, 2048, 128),
    ("large", 4096, 4096, 128),
    ("XL", 8192, 4096, 256),
]

DENSITIES = [0.01, 0.05, 0.10, 0.20, 0.30]  
SPARSITIES = [0.99, 0.95, 0.90, 0.80, 0.70] 
assert len(DENSITIES) == len(SPARSITIES)


# -------------------------------------------------------------
# helpers
# -------------------------------------------------------------
def initialize_matrices(M, N, K, sparsity):
    density = 1 - sparsity
    A = (torch.rand(M, K, device=DEVICE) < density).float() * torch.rand(M, K, device=DEVICE)
    B = torch.rand(K, N, device=DEVICE)
    C = torch.zeros(M, N, device=DEVICE)
    A_csr = A.to_sparse_csr()
    return A, B, C, A_csr


def preprocess_csr(A_csr):
    row = A_csr.crow_indices().to(torch.int32)
    col = A_csr.col_indices().to(torch.int32)
    val = A_csr.values()
    max_row_nnz = int((row[1:] - row[:-1]).max().item())
    return row, col, val, max_row_nnz


def time_fn(fn, *args, n_warmup=20, iters=100):
    for _ in range(n_warmup):
        fn(*args)
        torch.cuda.synchronize()

    t_list = []
    for _ in range(iters):
        t0 = time.time()
        fn(*args)
        torch.cuda.synchronize()
        t_list.append(time.time() - t0)

    return np.mean(t_list)


def count_gflops(nnz, N, time_s):
    flops = 2 * nnz * N
    return flops / time_s / 1e9


# -------------------------------------------------------------
# Kernels
# -------------------------------------------------------------
def fn_torch_mm(alpha, A_csr, beta, B, C):
    return alpha * torch.sparse.mm(A_csr, B) + beta * C

def fn_torch_addmm(alpha, A_csr, beta, B, C):
    return torch.sparse.addmm(C, A_csr, B, beta=beta, alpha=alpha)

def fn_naive(alpha, row, col, val, B, beta, C):
    return torch_csrspmm.csrspmm_naive(row, col, val, B, alpha, beta)

def fn_warp_per_row(alpha, row, col, val, B, beta, C):
    return torch_csrspmm.csrspmm_warp_per_row(row, col, val, B, alpha, beta)

def fn_warp_per_row_smem(alpha, row, col, val, max_row_nnz, B, beta, C):
    return torch_csrspmm.csrspmm_warp_per_row_smem(row, col, val, max_row_nnz, B, alpha, beta)

def fn_warp_per_row_fp4(alpha, row, col, val, B, beta, C):
    return torch_csrspmm.csrspmm_warp_per_row_fp4(row, col, val, B, alpha, beta)

def fn_warp_per_row_smem_fp4(alpha, row, col, val, max_row_nnz, B, beta, C):
    return torch_csrspmm.csrspmm_warp_per_row_smem_fp4(row, col, val, max_row_nnz, B, alpha, beta)

def main():
    alpha = 1.1
    beta = 0.5

    results = []

    for (case, M, N, K) in CONFIGS:
        for density, sparsity in zip(DENSITIES, SPARSITIES):

            print(f"\n===== {case} M={M} N={N} K={K} density={density:.2f} (sparsity={sparsity:.2f}) =====")

            A, B, C, A_csr = initialize_matrices(M, N, K, sparsity)
            nnz = A_csr.values().numel()

            row, col, val, max_row_nnz = preprocess_csr(A_csr)

            timings = {
                "torch.mm": time_fn(fn_torch_mm, alpha, A_csr, beta, B, C),
                "torch.addmm": time_fn(fn_torch_addmm, alpha, A_csr, beta, B, C),
                "Naive": time_fn(fn_naive, alpha, row, col, val, B, beta, C),
                "WarpPerRow": time_fn(fn_warp_per_row, alpha, row, col, val, B, beta, C),
                "WarpPerRowSmem": time_fn(fn_warp_per_row_smem, alpha, row, col, val, max_row_nnz, B, beta, C),
                "WarpPerRowFp4": time_fn(fn_warp_per_row_fp4, alpha, row, col, val, B, beta, C),
                "WarpPerRowSmemFp4": time_fn(fn_warp_per_row_smem_fp4, alpha, row, col, val, max_row_nnz, B, beta, C),
            }

            for algo, t in timings.items():
                gflops = count_gflops(nnz, N, t)

                results.append({
                    "case": case,
                    "M": M, "N": N, "K": K,
                    "density": density,
                    "sparsity": sparsity,
                    "lib": "torch" if "torch" in algo else "custom",
                    "algo": algo,
                    "gflops": gflops,
                })

                print(f"{algo:<18}: {gflops:8.2f} GFLOP/s")


    # save to .csv
    csv_path = "csrspmm_vs_torchsparse.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case","M","N","K","density","sparsity","lib","algo","gflops"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to: {csv_path}")


    # Visualization + Summary ----------------
    df = pd.DataFrame(results)

    sns.set(style="whitegrid", font_scale=1.2)

    for case in df["case"].unique():
        subset = df[df["case"] == case]

        plt.figure(figsize=(10, 6))

        sns.lineplot(
            data=subset,
            x="density",
            y="gflops",
            hue="algo",
            marker="o",
            linewidth=2,
            alpha=0.8
        )

        plt.title(f"SpMM Performance for Size: {case}")
        plt.xlabel("Density")
        plt.ylabel("GFLOPS")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        handles, labels = plt.gca().get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        plt.legend(unique.values(), unique.keys(), title="Algorithm")

        outfile = f"torchcsrspmm_{case}.png"
        plt.savefig(outfile, dpi=300)
        plt.close()

    print("\n========= Summary =========")

    mean_gflops = df.groupby("algo")["gflops"].mean().rename("mean_gflops")

    cus = df[df["algo"] == "torch.mm"][["case", "density", "gflops"]]
    cus = cus.rename(columns={"gflops": "cusparse_gflops"})

    merged = df.merge(cus, on=["case", "density"], how="left")
    merged["speedup"] = merged["gflops"] / merged["cusparse_gflops"]

    mean_speedup = merged.groupby("algo")["speedup"].mean().rename("mean_speedup")

    summary = pd.concat([mean_gflops, mean_speedup], axis=1)
    summary = summary.sort_values("mean_gflops", ascending=False)

    print(summary.to_string())


if __name__ == "__main__":
    main()

