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
    str(PROJECT_ROOT / "torch_csrspmm" / "binding.cpp"),
    str(PROJECT_ROOT / "torch_csrspmm" / "torch_csrspmm_naive.cpp"),
    str(PROJECT_ROOT / "torch_csrspmm" / "torch_csrspmm_warp_per_row.cpp"),
    str(PROJECT_ROOT / "torch_csrspmm" / "torch_csrspmm_warp_per_row_smem.cpp"),
    str(PROJECT_ROOT / "torch_csrspmm" / "torch_csrspmm_warp_per_row_fp4.cpp"),
    str(PROJECT_ROOT / "torch_csrspmm" / "torch_csrspmm_warp_per_row_smem_fp4.cpp"),

    # Host utilities
    str(PROJECT_ROOT / "src" / "csrspmm.cpp"),
    str(PROJECT_ROOT / "src" / "host" / "csr_utils.cpp"),
    str(PROJECT_ROOT / "src" / "host" / "dense_utils.cpp"),
    str(PROJECT_ROOT / "src" / "host" / "spmm_launcher.cu"),

    # Device kernels
    str(PROJECT_ROOT / "src" / "device" / "csrspmm_naive.cu"),
    str(PROJECT_ROOT / "src" / "device" / "csrspmm_naive_smem.cu"),
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
        "-march=native",
        "-ffast-math",
    ],
    "nvcc": [
        "-O3",
        "--use_fast_math",
        "--expt-relaxed-constexpr",

        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",

        # Architectures
        "-gencode=arch=compute_70,code=sm_70",
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_90,code=sm_90",
    ],
}

# --------------------------------------------------
# CUDA Extension definition
# --------------------------------------------------
ext_modules = [
    CUDAExtension(
        name="torch_csrspmm._C",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
    )
]

# --------------------------------------------------
# Setup
# --------------------------------------------------
setup(
    name="torch_csrspmm",
    version="0.0.1",
    packages=["torch_csrspmm"], 
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
