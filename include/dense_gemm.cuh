#pragma once

void dense_gemm_cpu(
    int M, int N, int K,
    float alpha, float* A, float* B, 
    float beta, float* C);

