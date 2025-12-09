# CSRSPMM: Optimized CUDA Kernels for CSR Sparse x Dense Matrix Multiplication
**CSRSPMM** is a high-performance library for multiplying Compressed Sparse Row (CSR) matrices with dense matrices on NVIDIA GPUs. It includes a generic CUDA backend, and a PyTorch extension for easy integration into deep learning workflows.

## 1. Project Structure

```
.
├── benchmarks/          # Benchmarking scripts
├── common/              # Common CUDA helper utilities
├── include/             # Public C++ headers for the library
├── results/             # Plots, CSV logs, and Nsight profiles
├── src/                 # Source C++/CUDA implementation (host + device)
├── tests/               # Unit tests for C++, CUDA, and PyTorch bindings
├── torch_csrspmm/       # PyTorch extension front-end + Python API
├── CMakeLists.txt       # Build system
├── setup.py             # Python packaging for PyTorch extension
└── README.md
```
**Note**: The project structure may evolve as more kernels and features are added.


## 2. Requirements
**CUDA**
- CUDA Toolkit 12.0+ (tested heavily with CUDA 13.0)
- Latest NVIDIA driver matching your toolkit
- GPU with compute capability 7.0+ (Turing, Ampere, Ada, Hopper recommended)

**Python** (for PyTorch benchmarks)
- Python 3.10+
- PyTorch with CUDA support (follow the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/))
- These following packages are also requires for plotting and utilities: `matplotlib`, `pandas`, `numpy`, `seaborn`, `setuptool`.

To install our `torch_csrspmm` extension to you environment, run this (from the project root)
```
pip install -e .
```

## 3. Build Instructions
From the project root:
```
mkdir build
cd build
cmake ..
make -j
```
Note that you can adjust the compiler flags and target architecture in `CMakeLists.txt` to fit your machine.

## 4. Running Benchmarks
### 4.1 Compare against `cuSPARSE`:
```
cd build/
python ../benchmarks/csrspmm_vs_cusparse.py
```

### 4.2 Compare against `torch.sparse`
```
python torch_csrspmm/torch_benchmarks.py
```

### 4.3 Profiling kernels with Nsight Compute
To generate `.ncu-rep` reports for each kernel 
```
cd build/
bash profile_kernels.sh
```
This step requires:
- `ncu` (Nsight Compute) installed
- Proper user permissions to access GPU performance counters (on some systems, this may require admin/root or enabling developer mode)

### Notes
- Results from our experiments (on a NVIDIA GeForce RTX4070) are save in `./results/`.
- Code for unit tests and benchmarks are written in `./tests/` and `./benchmarks/`. Compile it using `CMakeLists.txt` and run with `-h` for further instructions.
