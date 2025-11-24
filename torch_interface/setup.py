from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# -------------------------------------------------------------------
# Project root = parent of torch_interface
# -------------------------------------------------------------------
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent
print("PROJECT ROOT =", PROJECT_ROOT)

# print(">>> PROJECT ROOT =", PROJECT_ROOT)

# -------------------------------------------------------------------
# Source files for the extension
# -------------------------------------------------------------------
sources = [
    # PyTorch binding wrappers
    str(PROJECT_ROOT / "torch_interface" / "binding.cpp"),
    str(PROJECT_ROOT / "torch_interface" / "naive_wrapper.cpp"),

    # Host C++ sources
    str(PROJECT_ROOT / "src" / "host" / "csr_utils.cpp"),
    str(PROJECT_ROOT / "src" / "host" / "dense_utils.cpp"),
    str(PROJECT_ROOT / "src" / "host" / "spmm_launcher.cu"),

    # Device CUDA kernels
    str(PROJECT_ROOT / "src" / "device" / "csrspmm_naive.cu"),
]

# -------------------------------------------------------------------
# Include directories so <csrspmm/*.h> resolve correctly
# -------------------------------------------------------------------
include_dirs = [
    str(PROJECT_ROOT / "include"),             # include/csrspmm/*.h
    str(PROJECT_ROOT / "src" / "host"),
    str(PROJECT_ROOT / "src" / "device"),
]

# -------------------------------------------------------------------
# Compiler flags for MSVC + CUDA 13 + PyTorch
# -------------------------------------------------------------------
extra_compile_args = {
    "cxx": [
        "/O2",        # MSVC optimization
        "/std:c++17",
    ],
    "nvcc": [
        "-O3",
        "--expt-relaxed-constexpr",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",

        # VLAB / Turing (Sm75)
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_75,code=compute_75",
    ],
}

# -------------------------------------------------------------------
# Extension module
# -------------------------------------------------------------------
ext_modules = [
    CUDAExtension(
        name="csrspmm_torch",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
    )
]

# -------------------------------------------------------------------
# Install package
# -------------------------------------------------------------------
setup(
    name="csrspmm_torch",
    version="0.0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    packages=["torch_interface"],
)
