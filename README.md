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
├── requirements.txt
├── Makefile                    # Build automation for kernels, tests, and benchmarks
└── README.md
```
NOTE: project structure is subjected to change


## Requirements and Setup
To setup Pytorch (conda virtual environment recommended)
```
pip install -r requirements.txt
```

## Usage
```
mkdir build
cd build
cmake ..
make -j
```

You can add `-DCMAKE_BUILD_TYPE=Debug` to `cmake` command to compile with debug flags. <br>
Then run unit test. Example:
```
./test_spmm_csr -h
```

## TODO
- Profile kernels
- cuSPARSELt
