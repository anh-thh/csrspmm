#include <csrspmm/matrix.h>
#include <csrspmm/error_check.h>
#include <csrspmm/csr_utils.h>

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <algorithm>

namespace csrspmm {

void csr_alloc_host(CSRMatrix& csr, int height, int width, int nnz)
{
    if (csr.row_ptr || csr.col_idx || csr.values) {
        std::cerr << "csr_alloc_host error: already allocated host memory.\n";
        std::exit(1);
    }

    csr.height = height;
    csr.width  = width;
    csr.nnz    = nnz;

    csr.row_ptr = static_cast<int*>(std::malloc((height + 1) * sizeof(int)));
    csr.col_idx = static_cast<int*>(std::malloc(nnz * sizeof(int)));
    csr.values  = static_cast<float*>(std::malloc(nnz * sizeof(float)));

    if (!csr.row_ptr || !csr.col_idx || !csr.values) {
        std::cerr << "csr_alloc_host: malloc failed\n";
        std::exit(1);
    }
}


void csr_free_host(CSRMatrix& csr)
{
    if (csr.row_ptr) { std::free(csr.row_ptr); csr.row_ptr = nullptr; }
    if (csr.col_idx) { std::free(csr.col_idx); csr.col_idx = nullptr; }
    if (csr.values)  { std::free(csr.values);  csr.values  = nullptr; }

    csr.height = csr.width = csr.nnz = csr.max_row_nnz = 0;
}

void csr_alloc_device(CSRMatrix& csr, int height, int width, int nnz)
{
    if (csr.row_ptr || csr.col_idx || csr.values) {
        std::cerr << "csr_alloc_device error: already allocated device memory.\n";
        std::exit(1);
    }

    csr.height = height;
    csr.width  = width;
    csr.nnz    = nnz;

    cudaCheck(cudaMalloc(&csr.row_ptr, (height + 1) * sizeof(int)));
    cudaCheck(cudaMalloc(&csr.col_idx, nnz * sizeof(int)));
    cudaCheck(cudaMalloc(&csr.values,  nnz * sizeof(float)));
}

void csr_free_device(CSRMatrix& csr)
{
    if (csr.row_ptr) { cudaCheck(cudaFree(csr.row_ptr)); csr.row_ptr = nullptr; }  
    if (csr.col_idx) { cudaCheck(cudaFree(csr.col_idx)); csr.col_idx = nullptr; }
    if (csr.values)  { cudaCheck(cudaFree(csr.values));  csr.values  = nullptr; }

    csr.height = csr.width = csr.nnz = csr.max_row_nnz = 0;
}


void dense2csr(const DenseMatrix& dense, CSRMatrix& csr, float tol)
{
    if (!dense.data || dense.height <= 0 || dense.width <= 0) {
        std::cerr << "dense2csr: invalid dense matrix\n";
        std::exit(1);
    }

    if (csr.row_ptr || csr.col_idx || csr.values) {
        std::cerr << "dense2csr error: destination must be null.\n";
        std::exit(1);
    }

    int H = dense.height;
    int W = dense.width;

    int nnz = 0;
    for (int i = 0; i < H * W; ++i)
        if (std::fabs(dense.data[i]) > tol) ++nnz;

    csr_alloc_host(csr, H, W, nnz);

    int idx = 0;
    csr.row_ptr[0] = 0;

    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            float v = dense.data[i * W + j];
            if (std::fabs(v) > tol) {
                csr.col_idx[idx] = j;
                csr.values[idx]  = v;
                idx++;
            }
        }
        csr.row_ptr[i + 1] = idx;
    }

    int mx = 0;
    for (int i = 0; i < H; i++)
        mx = std::max(mx, csr.row_ptr[i+1] - csr.row_ptr[i]);

    csr.max_row_nnz = mx;
}


void csr2dense(const CSRMatrix& csr, DenseMatrix& dense)
{
    if (!csr.row_ptr || !csr.col_idx || !csr.values ||
        csr.height <= 0 || csr.width <= 0) {
        std::cerr << "csr2dense: invalid csr\n";
        std::exit(1);
    }

    if (dense.data != nullptr) {
        std::cerr << "csr2dense error: destination must be null.\n";
        std::exit(1);
    }

    size_t bytes = size_t(csr.height) * csr.width * sizeof(float);
    dense.data = static_cast<float*>(std::malloc(bytes));

    if (!dense.data) {
        std::cerr << "csr2dense: malloc failed\n";
        std::exit(1);
    }

    dense.height = csr.height;
    dense.width  = csr.width;

    std::memset(dense.data, 0, bytes);

    for (int i = 0; i < csr.height; ++i) {
        for (int p = csr.row_ptr[i]; p < csr.row_ptr[i + 1]; ++p) {
            dense.data[i * csr.width + csr.col_idx[p]] = csr.values[p];
        }
    }
}



void print_csr(const CSRMatrix& csr)
{
    std::cout << "CSRMatrix (HOST): "
              << csr.height << "×" << csr.width
              << ", nnz=" << csr.nnz
              << ", max_row_nnz=" << csr.max_row_nnz << "\n";

    std::cout << "row_ptr: [ ";
    for (int i = 0; i <= csr.height; i++)
        std::cout << csr.row_ptr[i] << " ";
    std::cout << "]\n";

    std::cout << "col_idx: [ ";
    for (int i = 0; i < csr.nnz; i++)
        std::cout << csr.col_idx[i] << " ";
    std::cout << "]\n";

    std::cout << "values:  [ ";
    for (int i = 0; i < csr.nnz; i++)
        std::cout << csr.values[i] << " ";
    std::cout << "]\n";
}


float compression_ratio(const CSRMatrix& csr)
{
    int dense = csr.height * csr.width;
    int sparse = csr.nnz + csr.nnz + (csr.height + 1); 

    return float(dense) / float(sparse);
}



void csr_host2device(const CSRMatrix& hA, CSRMatrix& dA)
{
    if (!hA.row_ptr || !hA.col_idx || !hA.values) {
        std::cerr << "csr_host2device: host CSRMatrix invalid\n";
        std::exit(1);
    }

    if (dA.row_ptr || dA.col_idx || dA.values) {
        std::cerr << "csr_host2device error: destination must be null.\n";
        std::exit(1);
    }

    csr_alloc_device(dA, hA.height, hA.width, hA.nnz);
    dA.max_row_nnz = hA.max_row_nnz;

    cudaCheck(cudaMemcpy(dA.row_ptr, hA.row_ptr,
                         (hA.height + 1) * sizeof(int),
                         cudaMemcpyHostToDevice));

    cudaCheck(cudaMemcpy(dA.col_idx, hA.col_idx,
                         hA.nnz * sizeof(int),
                         cudaMemcpyHostToDevice));

    cudaCheck(cudaMemcpy(dA.values, hA.values,
                         hA.nnz * sizeof(float),
                         cudaMemcpyHostToDevice));
}


void csr_device2host(const CSRMatrix& dA, CSRMatrix& hA)
{
    if (!dA.row_ptr || !dA.col_idx || !dA.values) {
        std::cerr << "csr_device2host: device pointer invalid\n";
        std::exit(1);
    }

    if (hA.row_ptr || hA.col_idx || hA.values) {
        std::cerr << "csr_device2host error: destination must be null.\n";
        std::exit(1);
    }

    csr_alloc_host(hA, dA.height, dA.width, dA.nnz);
    hA.max_row_nnz = dA.max_row_nnz;

    cudaCheck(cudaMemcpy(hA.row_ptr, dA.row_ptr,
                         (dA.height + 1) * sizeof(int),
                         cudaMemcpyDeviceToHost));

    cudaCheck(cudaMemcpy(hA.col_idx, dA.col_idx,
                         dA.nnz * sizeof(int),
                         cudaMemcpyDeviceToHost));

    cudaCheck(cudaMemcpy(hA.values, dA.values,
                         dA.nnz * sizeof(float),
                         cudaMemcpyDeviceToHost));
}

} // namespace csrspmm

