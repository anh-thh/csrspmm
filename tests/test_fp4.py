import torch
import torch_csrspmm

M, K, N = 1024, 1024, 512
alpha, beta = 1.0, 0.0

# Random dense → then sparsified → CSR
A = torch.rand(M, K, device="cuda")
A = A * (torch.rand_like(A) > 0.8)
A_csr = A.to_sparse_csr()

B = torch.rand(K, N, device="cuda")

# Extract CSR
row = A_csr.crow_indices().to(torch.int32)
col = A_csr.col_indices().to(torch.int32)
val = A_csr.values().to(torch.float32)

# Run kernel
C_custom = torch_csrspmm._C.csrspmm_warp_per_row_fp4(row, col, val, B, alpha, beta)

# Compare with PyTorch ref
C_ref = torch.sparse.mm(A_csr, B)

max_diff = (C_custom - C_ref).abs().max().item()
print("Max difference:", max_diff)
