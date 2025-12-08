#pragma once

#include <csrspmm/matrix.h>
#include <csrspmm/config.h>

namespace csrspmm {

void launch_naive(const CSRMatrix& A,
                  const DenseMatrix& B,
                  DenseMatrix& C,
                  float alpha,
                  float beta);
            
void launch_naive_shared(const CSRMatrix& A,
                  const DenseMatrix& B,
                  DenseMatrix& C,
                  float alpha,
                  float beta);

void launch_warp_per_row(const CSRMatrix& A,
                         const DenseMatrix& B,
                         DenseMatrix& C,
                         float alpha,
                         float beta);

void launch_warp_per_row_fp4(const CSRMatrix& A,
                              const DenseMatrix& B,
                              DenseMatrix& C,
                              float alpha,
                              float beta);

void launch_warp_per_row_smem(const CSRMatrix& A,
                                  const DenseMatrix& B,
                                  DenseMatrix& C,
                                  float alpha,
                                  float beta);


void launch_warp_per_row_smem_fp4(const CSRMatrix& A,
                                  const DenseMatrix& B,
                                  DenseMatrix& C,
                                  float alpha,
                                  float beta);

} // namespace csrspmm

