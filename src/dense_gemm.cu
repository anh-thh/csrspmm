#include "dense_gemm.cuh"

#define ACCUMULATOR_TYPE double

void dense_gemm_cpu(
    int M, int N, int K,
    float alpha, float* A, float* B, 
    float beta, float* C)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            ACCUMULATOR_TYPE sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = static_cast<float>(alpha * sum + beta * C[i * N + j]);
        }
    }
}
