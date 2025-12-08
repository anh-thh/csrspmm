# CSRSPMM: CUDA Sparse x Dense Matrix Multiplication


## Project Structure

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


## Requirements
CUDA
- CUDA Toolkit 12.0+ (tested heavily with CUDA 13.0)
- Latest NVIDIA driver matching your toolkit
- GPU with compute capability 7.0+ (Turing, Ampere, Ada, Hopper recommended)

Python (for PyTorch benchmarks)
- Python 3.10+
- PyTorch with CUDA support (follow the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/))


## Build Instructions
From the project root:
```
mkdir build
cd build
cmake ..
make -j
```

## Running Benchmarks
### Compare Against cuSPARSE:
```
cd build/
python ../benchmarks/csrspmm_vs_cusparse.py
```

### Compare Against `torch.sparse`
First install the PyTorch extension
```
pip install -e .
```
Run 
```
python torch_csrspmm/torch_benchmarks.py
```

### Profiling Kernels with Nsight Compute
To generate `.ncu-rep` reports for each kernel
```
cd build/
bash profile_kernels.sh
```

### Notes
Results from our experiments are save in `./results/`.
