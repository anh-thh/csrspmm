#pragma once

void print_dense_matrix(const float* matrix, int num_rows, int num_cols);

void init_random_dense_matrix(
    float* matrix, int num_rows, int num_cols, 
    float low, float high, float sparsity,
    bool is_int = false
);

bool compare_dense_matrices(
    const float* mat_a, const float* mat_b, 
    int num_rows, int num_cols,
    float tol = 1e-5f
);
