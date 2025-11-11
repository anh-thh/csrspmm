#include "dense_utils.cuh"
#include <iostream>

void print_dense_matrix(const float* matrix, int num_rows, int num_cols) {
    std::cout << "Dense Matrix (" << num_rows << "x" << num_cols << "):" << std::endl;

    for (int i = 0; i < num_rows; ++i) {
        std::cout << "  [ ";
        for (int j = 0; j < num_cols; ++j) {
            std::cout << matrix[i * num_cols + j];
            if (j < num_cols - 1)
                std::cout << ", ";
        }
        std::cout << " ]" << std::endl;
    }
    std::cout << std::endl;
}


void init_random_dense_matrix(
    float* matrix, int num_rows, int num_cols, 
    float low, float high, float sparsity,
    bool is_int
) {
    for (int i = 0; i < num_rows * num_cols; ++i) {
        float rand_val = static_cast<float>(rand()) / RAND_MAX;
        if (rand_val < sparsity) {
            matrix[i] = 0.0f; 
        } else {
            float val = static_cast<float>(rand()) / RAND_MAX * (high - low) + low;
            matrix[i] = is_int ? static_cast<int>(val) : val;
        }
    }
}

bool compare_dense_matrices(
    const float* mat_a, const float* mat_b, 
    int num_rows, int num_cols,
    float tol
) {
    for (int i = 0; i < num_rows * num_cols; ++i) {
        if (fabs(mat_a[i] - mat_b[i]) > tol) {
            std::cout << "Mismatch at index " << i << ": " 
                      << mat_a[i] << " vs " << mat_b[i] << std::endl;
            return false;
        }
    }
    return true;
}
