#!/bin/bash
set -e  # stop on first error

echo "======================================"
echo "   Rebuilding csrspmm_torch extension"
echo "======================================"

# Activate your conda env
echo "Activating conda env..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spmm_env

# Clean pip cache (optional but recommended)
echo "Purging pip cache..."
pip cache purge

# Rebuild PyTorch extension
echo "Installing in editable mode..."
python -m pip install --no-build-isolation -e .

echo ""
echo "======================================"
echo "        Running Torch Benchmark"
echo "======================================"

python torch_interface/torch_benchmarks.py

echo ""
echo "======================================"
echo "          Benchmark Complete"
echo "======================================"
