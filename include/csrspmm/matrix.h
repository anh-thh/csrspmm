#pragma once
#include <cuda_runtime.h>

namespace csrspmm {

struct CSRMatrix {
    int height = 0;      // number of rows 
    int width = 0;       // number of cols
    int nnz = 0;         // number of nonzeros

    int* row_ptr = nullptr;     // device pointer, size = height+1
    int* col_idx = nullptr;     // device pointer, size = nnz
    float* values = nullptr;   // device pointer, size = nnz
    
    int max_row_nnz = 0;
};


struct DenseMatrix {
    int height = 0;      // rows
    int width = 0;       // cols
    float* data = nullptr;     // device pointer, length = height * width
};

} // namespace csrspmm

