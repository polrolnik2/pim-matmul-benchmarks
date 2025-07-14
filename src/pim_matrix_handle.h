#ifndef __PIM_MATRIX_HANDLE_H___
#define __PIM_MATRIX_HANDLE_H___

/*
 * PIM Matrix Handle
 *
 * This structure represents a handle to a matrix in the PIM (Processing In Memory)
 * context. It contains metadata about the matrix and its submatrices as they are stored in the PIM.
 */
typedef struct {
    char*   pim_handle;
    int16_t submatrix_rows; ///< Number of rows in the submatrix
    int16_t submatrix_cols; ///< Number of columns in the submatrix
} pim_matrix_handle_t;

pim_matrix_handle_t* broadcast_matrix_to_pim(const Matrix* matrix, int16_t submatrix_rows, int16_t submatrix_cols);

pim_matrix_handle_t* scatter_matrix_to_pim(const Matrix* matrix, int16_t submatrix_rows, int16_t submatrix_cols);

Matrix* gather_matrix_from_pim(const char *handle, int16_t rows, int16_t cols);

void free_pim_matrix_handle(pim_matrix_handle_t* handle);

pim_matrix_handle_t* add_pim_matrices(const pim_matrix_handle_t* handle1, const pim_matrix_handle_t* handle2);

pim_matrix_handle_t* multiply_pim_matrices(const pim_matrix_handle_t* handle1, const pim_matrix_handle_t* handle2);

#endif // __PIM_MATRIX_HANDLE_H___