#include <stdexcept>

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

    kernel::csrspmm_naive<<<grid, block>>>(
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

void launch_naive_shared(const CSRMatrix& A,
                         const DenseMatrix& B,
                         DenseMatrix& C,
                         float alpha,
                         float beta)
{
    // One block per row, threads over columns
    dim3 block(128, 1);
    dim3 grid((B.width + block.x - 1) / block.x,
              A.height);

    kernel::csrspmm_naive_shared<<<grid, block>>>(
        A.height,   // M
        A.width,    // N
        B.width,    // K
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

    kernel::csrspmm_warp_per_row<<<num_blocks, threads_per_block>>>(
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

void launch_warp_per_row_fp4(const CSRMatrix& A,
                              const DenseMatrix& B,
                              DenseMatrix& C,
                              float alpha,
                              float beta)
{
    const int warps_per_block   = 4;
    const int threads_per_block = warps_per_block * WARP_SIZE;

    int num_blocks = (A.height + warps_per_block - 1) / warps_per_block;

    kernel::csrspmm_warp_per_row_fp4<<<num_blocks, threads_per_block>>>(
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


void launch_warp_per_row_smem(const CSRMatrix& A,
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

    kernel::csrspmm_warp_per_row_smem<<<
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


void launch_warp_per_row_smem_fp4(const CSRMatrix& A,
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

    kernel::csrspmm_warp_per_row_smem_fp4<<<
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
