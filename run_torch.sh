# -----------------------------
# ENVIRONMENT SETUP
# -----------------------------

# Create fresh env (mac/linux)
python3.11 -m venv spmm_env
source spmm_env/bin/activate

# or (Windows PowerShell)
# python -m venv spmm_env
# spmm_env\Scripts\activate

pip install --upgrade pip setuptools wheel

# -----------------------------
# TORCH INSTALL
# -----------------------------

# (A) macOS or CPU-only machine:
pip install torch torchvision torchaudio

# (B) Windows/Linux with CUDA 13 (like Biglab):
# pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# (C) Windows/Linux with CUDA 12.1:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# -----------------------------
# INSTALL SPMM EXTENSION
# -----------------------------
cd torch_interface
pip install -e .
cd ..

# -----------------------------
# RUN TEST
# -----------------------------
python tests/test_naive_spmm.py
