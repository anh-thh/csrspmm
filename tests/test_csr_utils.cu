#include "csr_utils.cuh"
#include "dense_utils.cuh"
#include "helper.cuh"
#include <cstdlib>
 

int main() {
    const int num_rows = 1000;
    const int num_cols = 2000;
    float* dense_matrix = (float*) malloc(num_rows * num_cols * sizeof(float));
    init_random_dense_matrix(dense_matrix, num_rows, num_cols, -10.0f, 10.0f, 0.7f, true);
    // print_dense_matrix(dense_matrix, num_rows, num_cols);

    CSRMatrix csr_matrix;
    dense2csr(dense_matrix, num_rows, num_cols, csr_matrix);
    // print_csr_matrix(csr_matrix);

    float* recovered_dense = (float*) malloc(num_rows * num_cols * sizeof(float));
    csr2dense(csr_matrix, recovered_dense);
    // print_dense_matrix(recovered_dense, num_rows, num_cols);
    

    bool are_equal = compare_dense_matrices(dense_matrix, recovered_dense, num_rows, num_cols);
    if (are_equal) {
        std::cout << "Pass: The original and recovered dense matrices are equal." << std::endl;
    } else {
        std::cout << "Error: The original and recovered dense matrices are NOT equal." << std::endl;
    }

    free(dense_matrix);
    free(recovered_dense);
    return 0;
}
