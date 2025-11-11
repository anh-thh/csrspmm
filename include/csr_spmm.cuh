#pragma once
#include "csr_utils.cuh"
#include <cuda_runtime.h>
#include <string>
#include <iostream>
#define ACCUMULATOR_TYPE float


enum Algo {
    cuSPARSELt = 0,
    naive,
    warp_per_row,
    numAlgos
};


inline Algo parse_csr_algo(const std::string& name) {
    if (name == "cuSPARSELt") return cuSPARSELt;
    if (name == "naive")      return naive;
    if (name == "warp_per_row") return warp_per_row;

    std::cerr << "Error: unknown algorithm name '" << name << "'\n";
    std::exit(1);
}



// A[M,N] x B[N, K] => C[M, K]
void run_csr_spmm(Algo algo,
                  int M, int N, int K,
                  float alpha, float beta,
                  float* A_values,
                  int* A_col_idx,
                  int* A_row_ptr,
                  float *B, 
                  float *C);

__global__ void csr_spmm_naive(
    int M, int N, int K,
    float alpha, 
    float beta, 
    const float* __restrict__ A_values, 
    const int* __restrict__ A_col_idx, 
    const int* __restrict__ A_row_ptr, 
    const float* __restrict__ B, 
    float* __restrict__ C);


#define WARP_SIZE 32
__global__ void csr_spmm_warp_per_row(
    int M, int N, int K,
    float alpha, float beta,
    const float* __restrict__ A_values,
    const int*  __restrict__ A_col_idx,
    const int*  __restrict__ A_row_ptr,
    const float* __restrict__ B,
    float* __restrict__ C);
