#include <csrspmm/matrix.h>
#include <csrspmm/config.h>
#include <csrspmm/kernels.h>


namespace csrspmm {

void launch_naive(const CSRMatrix& A,
                  const DenseMatrix& B,
                  DenseMatrix& C,
                  float alpha,
                  float beta)
{
    dim3 block(32, 32);
    dim3 grid((B.width  + block.x - 1) / block.x,
              (A.height + block.y - 1) / block.y);

    kernel::csr_spmm_naive<<<grid, block>>>(
        A.height,
        A.width,
        B.width,
        alpha, beta,
        A.values,
        A.col_idx,
        A.row_ptr,
        B.data,
        C.data
    );
}

void launch_warp_per_row(const CSRMatrix& A,
                         const DenseMatrix& B,
                         DenseMatrix& C,
                         float alpha,
                         float beta)
{
    const int warps_per_block   = 4;
    const int threads_per_block = warps_per_block * WARP_SIZE;

    int num_blocks = (A.height + warps_per_block - 1) / warps_per_block;

    kernel::csr_spmm_warp_per_row<<<num_blocks, threads_per_block>>>(
        A.height,
        A.width,
        B.width,
        alpha, beta,
        A.values,
        A.col_idx,
        A.row_ptr,
        B.data,
        C.data
    );
}

void launch_warp_per_row_vec4(const CSRMatrix& A,
                              const DenseMatrix& B,
                              DenseMatrix& C,
                              float alpha,
                              float beta)
{
    const int warps_per_block   = 4;
    const int threads_per_block = warps_per_block * WARP_SIZE;

    int num_blocks = (A.height + warps_per_block - 1) / warps_per_block;

    kernel::csr_spmm_warp_per_row_vec4<<<num_blocks, threads_per_block>>>(
        A.height,
        A.width,
        B.width,
        alpha, beta,
        A.values,
        A.col_idx,
        A.row_ptr,
        B.data,
        C.data
    );
}


void launch_warp_per_row_sharemem(const CSRMatrix& A,
                                  const DenseMatrix& B,
                                  DenseMatrix& C,
                                  float alpha,
                                  float beta)
{
    const int warps_per_block   = 4;
    const int threads_per_block = warps_per_block * WARP_SIZE;

    size_t shmem_size =
        (size_t)warps_per_block *
        (size_t)A.max_row_nnz *
        (sizeof(float) + sizeof(int));

    int num_blocks = (A.height + warps_per_block - 1) / warps_per_block;

    kernel::csr_spmm_warp_per_row_sharemem<<<
        num_blocks,
        threads_per_block,
        shmem_size>>>(
        A.height,
        A.width,
        B.width,
        alpha, beta,
        A.values,
        A.col_idx,
        A.row_ptr,
        A.max_row_nnz,
        B.data,
        C.data
    );
}


}
