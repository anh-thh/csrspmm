#pragma once
#include <csrspmm/matrix.h>


namespace csrspmm {

void dense_alloc_host(DenseMatrix& A, int height, int width);
void dense_alloc_device(DenseMatrix& A, int height, int width);

void dense_free_host(DenseMatrix& A);
void dense_free_device(DenseMatrix& A);

void dense_host2device(const DenseMatrix& hA, DenseMatrix& dA);
void dense_device2host(const DenseMatrix& dA, DenseMatrix& hA);

void dense_init_random(DenseMatrix& A,
                       float low = -1.0f,
                       float hight =  1.0f,
                       float sparsity = 0.0f,
                       bool integer_values = false);

void print_dense(const DenseMatrix& A);

bool dense_compare(const DenseMatrix& A,
                   const DenseMatrix& B,
                   float tol = 1e-4f);

void dense_gemm_cpu(int M, int N, int K,
                    float alpha, float beta,
                    const DenseMatrix& A,
                    const DenseMatrix& B,
                    DenseMatrix& C);

}
