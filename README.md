# CUDA Sparse x Dense Matrix Multiplication (CSR x Dense)


## Project Structure

```
├── benchmarks/                 # Benchmark programs and profiling
│ └── bench_spmm_csr.cu 
│
├── include/                    # Header
│ ├── csr_spmm.cuh
│ ├── csr_utils.cuh 
│ ├── dense_gemm.cuh
│ ├── dense_utils.cuh 
│ └── helper.cuh
│
├── src/                        # Source
│ ├── csr_spmm.cu 
│ ├── csr_utils.cu
│ ├── dense_gemm.cu 
│ └── dense_utils.cu 
│
├── tests/                      # Unit and correctness tests
│ ├── test_csr_utils.cu 
│ └── test_spmm_csr.cu
│
├── torch_interface/
│
├── Makefile                    # Build automation for kernels, tests, and benchmarks
└── README.md
```
NOTE: project structure is subjected to change


## Usage

```
make test_csr_utils
```
then 
```
./test_csr_utils
```

