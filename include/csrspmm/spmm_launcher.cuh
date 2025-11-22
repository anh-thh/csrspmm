#pragma once

#include <csrspmm/matrix.h>
#include <csrspmm/config.h>

namespace csrspmm {

void launch_naive(const CSRMatrix& A,
                  const DenseMatrix& B,
                  DenseMatrix& C,
                  float alpha,
                  float beta);

void launch_warp_per_row(const CSRMatrix& A,
                         const DenseMatrix& B,
                         DenseMatrix& C,
                         float alpha,
                         float beta);

void launch_warp_per_row_vec4(const CSRMatrix& A,
                              const DenseMatrix& B,
                              DenseMatrix& C,
                              float alpha,
                              float beta);

void launch_warp_per_row_sharemem(const CSRMatrix& A,
                                  const DenseMatrix& B,
                                  DenseMatrix& C,
                                  float alpha,
                                  float beta);

} // namespace csrspmm

