#include "dense_gemm.cuh"

#define ACCUMULATOR_TYPE double

/*
 * Dense Gemm
 * A[M, N] x B[N, K] => C[M, K]
 */


void dense_gemm_cpu(
    int M, int N, int K,
    float alpha, float beta, 
    const float* A, 
    const float* B, 
    float* C)
{
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float sum = 0.0f;
            for (int j = 0; j < N; ++j) {
                sum += A[i * N + j] * B[j * K + k];
            }
            C[i * K + k] = alpha * sum + beta * C[i * K + k];
        }
    }
}
