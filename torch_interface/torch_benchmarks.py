import torch
import torch_interface
import time

print(f"Pytorch {torch.__version__}\n")


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This benchmark requires a GPU.")
else: 
    # GPU model 
    torch.cuda.init()
    DEVICE = torch.device("cuda")
    print("Running Pytorch benchmark on ", torch.cuda.get_device_name(0))


def initialize_matrices(M, N, K, sparsity=0.7):
    A = torch.rand(M, K, device=DEVICE, dtype=torch.float32)
    mask = torch.rand_like(A) > sparsity  
    A = A * mask
    B = torch.rand(K, N, device=DEVICE, dtype=torch.float32)
    C = torch.zeros(M, N, device=DEVICE, dtype=torch.float32)
    A_csr = A.to_sparse_csr()
    return A, B, C, A_csr



def run_benchmark(fn, *args, n_warmup=10, iterations=100, **kwargs):
    """Run a benchmark function with warm-up and timing."""
    # Warm-up
    for _ in range(n_warmup):
        fn(*args, **kwargs)
        torch.cuda.synchronize()

    latencies = []
    for _ in range(iterations):
        start = time.time()
        fn(*args, **kwargs)
        torch.cuda.synchronize()
        latencies.append(time.time() - start)

    return latencies


def fn_torch_sparse_mm(alpha, A_csr, beta, B, C):
    return alpha * torch.sparse.mm(A_csr, B) + beta * C


def fn_torch_sparse_addmm(alpha, A_csr, beta, B, C):
    return torch.sparse.addmm(C, A_csr, B, beta=beta, alpha=alpha)


def fn_custom_csr_spmm(alpha, A_csr, beta, B, C):
    return torch_interface.csr_spmm(alpha, A_csr, beta, B, C)



#  def test():
#      A, B, C, A_csr = initialize_matrices(10, 10, 10, 0.7)
#
#      alpha, beta = 1.0, 0.5
#
#      C_1 = fn_torch_sparse_mm(alpha, A_csr, beta, B, C)
#      C_2 = fn_torch_sparse_addmm(alpha, A_csr, beta, B, C)
#
#
#      max_diff = (C_1 - C_2).abs().max().item()
#
#      tol = 1e-5
#      if max_diff < tol:
#          print("PASS: Results match within tolerance.")
#      else:
#          print("FAIL: Results differ beyond tolerance.")

    


 
if __name__ == "__main__":
    M, N, K = 1024, 1024, 1024
    alpha = 1.0
    beta = 0.0
    sparsity = 0.7
    n_warmup = 10
    iterations = 100

    # TODO: use argparse

    #  test()

    A, B, C, A_csr = initialize_matrices(M, N, K, sparsity)

    t_mm = run_benchmark(fn_torch_sparse_mm, alpha, A_csr, beta, B, C,
                         n_warmup=n_warmup, iterations=iterations)

    t_addmm = run_benchmark(fn_torch_sparse_addmm, alpha, A_csr, beta, B, C,
                            n_warmup=n_warmup, iterations=iterations)

    t_custom = run_benchmark(fn_custom_csr_spmm,alpha, A_csr, beta, B, C,
                             n_warmup=n_warmup, iterations=iterations)


    # process data, graph, etc.
