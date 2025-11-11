#pragma once

struct CSRMatrix {
    int num_rows;
    int num_cols;
    int nnz;            // number of non-zeros 

    float* values;      // pointer, length = nnz
    int*   col_idx;     // pointer, length = nnz
    int*   row_ptr;     // pointer, length = num_rows + 1
};


void dense2csr(
    const float* dense_matrix, int num_rows, int num_cols, 
    CSRMatrix& csr_matrix,
    float tol = 0.0f);

void csr2dense(const CSRMatrix& csr_matrix, float* dense_matrix);

void print_csr_matrix(const CSRMatrix& csr_matrix);
