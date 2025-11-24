from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# --------------------------------------------------
# PROJECT ROOT (directory containing setup.py)
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
print(">>> PROJECT ROOT =", PROJECT_ROOT)

# --------------------------------------------------
# SOURCES: all host + device kernels + wrappers
# --------------------------------------------------
sources = [
    # Torch bindings
    str(PROJECT_ROOT / "torch_interface" / "binding.cpp"),
    str(PROJECT_ROOT / "torch_interface" / "naive_wrapper.cpp"),

    # Host utilities
    str(PROJECT_ROOT / "src" / "host" / "csr_utils.cpp"),
    str(PROJECT_ROOT / "src" / "host" / "dense_utils.cpp"),
    str(PROJECT_ROOT / "src" / "host" / "spmm_launcher.cu"),

    # Device kernels
    str(PROJECT_ROOT / "src" / "device" / "csrspmm_naive.cu"),
    str(PROJECT_ROOT / "src" / "device" / "csrspmm_warp_per_row.cu"),
    str(PROJECT_ROOT / "src" / "device" / "csrspmm_warp_per_row_fp4.cu"),
    str(PROJECT_ROOT / "src" / "device" / "csrspmm_warp_per_row_smem.cu"),
    str(PROJECT_ROOT / "src" / "device" / "csrspmm_warp_per_row_smem_fp4.cu"),
]

# --------------------------------------------------
# INCLUDE DIRECTORIES
# --------------------------------------------------
include_dirs = [
    str(PROJECT_ROOT / "include"),
    str(PROJECT_ROOT / "src" / "host"),
    str(PROJECT_ROOT / "src" / "device"),
]

# --------------------------------------------------
# CUDA + CXX compile flags for Windows + CUDA 13.0
# --------------------------------------------------
extra_compile_args = {
    "cxx": [
        "-O3",
    ],
    "nvcc": [
        "-O3",
        "--expt-relaxed-constexpr",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",

        # VLAB GPUs usually = Turing (sm_75)
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_75,code=compute_75",
    ],
}

# --------------------------------------------------
# CUDA Extension definition
# --------------------------------------------------
ext_modules = [
    CUDAExtension(
        name="csrspmm_torch",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
    )
]

# --------------------------------------------------
# Setup
# --------------------------------------------------
setup(
    name="csrspmm_torch",
    version="0.0.1",
    packages=["torch_interface"],   # so Python can import torch_interface
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
