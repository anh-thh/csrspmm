#pragma once

void dense_gemm_cpu(
    int M, int N, int K,
    float alpha, float beta, 
    const float* A, 
    const float* B, 
    float* C);

