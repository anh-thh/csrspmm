#include <csrspmm/dense_utils.h>
#include <csrspmm/error_check.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cmath>


namespace csrspmm {

void dense_alloc_host(DenseMatrix& A, int height, int width) {
    if (A.data != nullptr) {
        std::cerr << "dense_alloc_host error: already allocated host memory.\n";
        std::exit(1);
    }

    A.height = height;
    A.width  = width;
    A.data   = (float*)std::malloc(height * width * sizeof(float));
}

void dense_free_host(DenseMatrix& A) {
    std::free(A.data);
    A.data = nullptr;
}

void dense_alloc_device(DenseMatrix& A, int height, int width) {
    if (A.data != nullptr) {
        std::cerr << "dense_alloc_host error: already allocated device memory.\n";
        std::exit(1);
    }

    A.height = height;
    A.width  = width;
    A.data   = nullptr;

    cudaCheck(cudaMalloc(&A.data, height * width * sizeof(float)));
}

void dense_free_device(DenseMatrix& A) {
    cudaCheck(cudaFree(A.data));
    A.data = nullptr;
}

void dense_init_random(DenseMatrix& A,
                       float low,
                       float high,
                       float sparsity,
                       bool integer_values)
{
    int total = A.height * A.width;

    for (int i = 0; i < total; ++i) {
        float r = float(rand()) / RAND_MAX;

        if (r < sparsity) {
            A.data[i] = 0.0f;
        } else {
            float val = float(rand()) / RAND_MAX * (high - low) + low;
            A.data[i] = integer_values ? float(int(val)) : val;
        }
    }
}


void print_dense(const DenseMatrix& A) {
    std::cout << "DenseMatrix (" << A.height << "x" << A.width << "):\n";

    for (int i = 0; i < A.height; ++i) {
        std::cout << "  [ ";
        for (int j = 0; j < A.width; ++j) {
            std::cout << A.data[i * A.width + j];
            if (j < A.width - 1)
                std::cout << ", ";
        }
        std::cout << " ]\n";
    }
    std::cout << std::endl;
}


bool dense_compare(const DenseMatrix& A,
                   const DenseMatrix& B,
                   float tol)
{
    if (A.height != B.height || A.width != B.width) {
        std::cout << "Dense compare failed: shape mismatch\n";
        return false;
    }

    int total = A.height * A.width;

    for (int i = 0; i < total; ++i) {
        float a = A.data[i];
        float b = B.data[i];

        if (std::fabs(a - b) > tol) {
            std::cout << "Mismatch at index " << i
                      << ": " << a << " vs " << b << "\n";
            return false;
        }
    }
    return true;
}


void dense_gemm_cpu(int M, int N, int K,
                    float alpha, float beta,
                    const DenseMatrix& A,
                    const DenseMatrix& B,
                    DenseMatrix& C)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {

            float sum = 0.0f;

            for (int k = 0; k < N; ++k) {
                sum += A.data[i*N + k] * B.data[k*K + j];
            }

            C.data[i*K + j] = alpha * sum + beta * C.data[i*K + j];
        }
    }
}

void dense_host2device(const DenseMatrix& hA, DenseMatrix& dA)
{
    if (!hA.data) {
        std::cerr << "dense_host2device: host data is null\n";
        std::exit(1);
    }

    if (dA.data != nullptr) {
        std::cerr << "dense_host2device error: Only accept null destination.\n";
        std::exit(1);
    }
    dense_alloc_device(dA, hA.height, hA.width);

    size_t bytes = size_t(hA.height) * hA.width * sizeof(float);
    cudaCheck(cudaMemcpy(dA.data, hA.data, bytes, cudaMemcpyHostToDevice));
}


void dense_device2host(const DenseMatrix& dA, DenseMatrix& hA)
{
    if (!dA.data) {
        std::cerr << "dense_device2host: device data is null\n";
        std::exit(1);
    }

    if (hA.data != nullptr) {
        std::cerr << "dense_device2host error: only accept null destination.\n";
        std::exit(1);
    }

    dense_alloc_host(hA, dA.height, dA.width);

    size_t bytes = size_t(dA.height) * dA.width * sizeof(float);
    cudaCheck(cudaMemcpy(hA.data, dA.data, bytes, cudaMemcpyDeviceToHost));
}

}
