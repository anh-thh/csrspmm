#pragma once
#include <csrspmm/matrix.h>

namespace csrspmm {

void csr_alloc_host(CSRMatrix& csr, int height, int width, int nnz);
void csr_free_host(CSRMatrix& csr);

void csr_alloc_device(CSRMatrix& csr, int height, int width, int nnz);
void csr_free_device(CSRMatrix& csr);

void dense2csr(const DenseMatrix& dense, CSRMatrix& csr, float tol = 0.0f);
void csr2dense(const CSRMatrix& csr, DenseMatrix& dense);
void print_csr(const CSRMatrix& csr);
float compression_ratio(const CSRMatrix& csr);

void csr_host2device(const CSRMatrix& hA, CSRMatrix& dA);
void csr_device2host(const CSRMatrix& dA, CSRMatrix& hA);

} // namespace csrspmm

