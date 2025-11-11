#include "csr_utils.cuh"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cmath>
#include <cassert>


void dense2csr(
    const float* dense_matrix, int num_rows, int num_cols, 
    CSRMatrix& csr_matrix,
    float tol) 
{
    csr_matrix.num_rows = num_rows;
    csr_matrix.num_cols = num_cols;

    // count non-zero elements
    int nnz = 0;
    for (int i = 0; i < num_rows * num_cols; ++i) {
        if (fabsf(dense_matrix[i]) > tol)
            ++nnz;
    }

    csr_matrix.nnz = nnz;

    // Allocate arrays
    csr_matrix.values = new float[nnz];
    csr_matrix.col_idx = new int[nnz];
    csr_matrix.row_ptr = new int[num_rows + 1];

    int nnz_index = 0;
    csr_matrix.row_ptr[0] = 0;

    // Fill CSR arrays
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            float val = dense_matrix[i * num_cols + j];
            if (fabsf(val) > tol) {
                csr_matrix.values[nnz_index] = val;
                csr_matrix.col_idx[nnz_index] = j;
                ++nnz_index;
            }
        }
        csr_matrix.row_ptr[i + 1] = nnz_index;
    }

    assert(nnz_index == nnz && "Mismatch in counted and filled non-zero elements in CSR conversion.");
}


void csr2dense(const CSRMatrix& csr_matrix, float* dense_matrix) {
    int rows = csr_matrix.num_rows;
    int cols = csr_matrix.num_cols;

    // Initialize dense matrix
    std::memset(dense_matrix, 0, rows * cols * sizeof(float));

    // Fill in non-zero values
    for (int i = 0; i < rows; ++i) {
        for (int j = csr_matrix.row_ptr[i]; j < csr_matrix.row_ptr[i + 1]; ++j) {
            int col = csr_matrix.col_idx[j];
            dense_matrix[i * cols + col] = csr_matrix.values[j];
        }
    }
}


void print_csr_matrix(const CSRMatrix& csr_matrix) {
    std::cout << "CSR Matrix:" << std::endl;
    std::cout << "  Num Rows: " << csr_matrix.num_rows
              << ", Num Cols: " << csr_matrix.num_cols
              << ", NNZ: " << csr_matrix.nnz << std::endl;

    std::cout << "  Values:   [ ";
    for (int i = 0; i < csr_matrix.nnz; ++i) {
        std::cout << csr_matrix.values[i];
        if (i < csr_matrix.nnz - 1)
            std::cout << ", ";
    }
    std::cout << " ]" << std::endl;

    std::cout << "  Col Idxs: [ ";
    for (int i = 0; i < csr_matrix.nnz; ++i) {
        std::cout << csr_matrix.col_idx[i];
        if (i < csr_matrix.nnz - 1)
            std::cout << ", ";
    }
    std::cout << " ]" << std::endl;

    std::cout << "  Row Ptrs: [ ";
    for (int i = 0; i <= csr_matrix.num_rows; ++i) {
        std::cout << csr_matrix.row_ptr[i];
        if (i < csr_matrix.num_rows)
            std::cout << ", ";
    }
    std::cout << " ]" << std::endl << std::endl;
}


float csr_compression_ratio(const CSRMatrix& csr_matrix) {
    int dense_size = csr_matrix.num_rows * csr_matrix.num_cols;
    int csr_size = csr_matrix.nnz + csr_matrix.nnz + (csr_matrix.num_rows + 1); // values + col_idx + row_ptr
    
    float compression_ratio = static_cast<float>(dense_size) / static_cast<float>(csr_size);

    return compression_ratio;
}
