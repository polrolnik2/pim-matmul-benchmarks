/**
 * @file pim_matrix_handle.h
 * @brief PIM Matrix Handle and related operations for Processing In Memory (PIM) context.
 *
 * This header defines the structure and functions for managing matrices in the PIM context.
 * It provides APIs for broadcasting, scattering, gathering, and performing operations on matrices stored in PIM.
 */
#ifndef __PIM_MATRIX_HANDLE_H___
#define __PIM_MATRIX_HANDLE_H___

/**
 * @struct pim_matrix_handle_t
 * @brief Handle to a matrix in the PIM context.
 *
 * Contains metadata about the matrix and its submatrices as stored in PIM.
 * @var pim_handle Pointer to PIM-specific handle or identifier.
 * @var submatrix_rows Number of rows in the submatrix.
 * @var submatrix_cols Number of columns in the submatrix.
 */
typedef struct {
    char*   pim_handle;      ///< PIM-specific handle or identifier
    int16_t submatrix_rows;  ///< Number of rows in the submatrix
    int16_t submatrix_cols;  ///< Number of columns in the submatrix
} pim_matrix_handle_t;

/**
 * @brief Broadcast a matrix to all PIM units.
 * @param matrix Pointer to the matrix to broadcast.
 * @return Pointer to a new PIM matrix handle.
 */
pim_matrix_handle_t* broadcast_matrix_to_pim(const Matrix* matrix, simplepim_management_t* management);

/**
 * @brief Scatter a matrix to PIM units as submatrices.
 * @param matrix Pointer to the matrix to scatter.
 * @param submatrix_rows Number of rows per submatrix.
 * @param submatrix_cols Number of columns per submatrix.
 * @return Pointer to a new PIM matrix handle.
 */
pim_matrix_handle_t* scatter_matrix_to_pim(const Matrix* matrix, int16_t submatrix_rows, int16_t submatrix_cols, simplepim_management_t* management);

/**
 * @brief Gather a matrix from PIM using a handle.
 * @param handle PIM handle string.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return Pointer to a new Matrix object.
 */
Matrix* gather_matrix_from_pim(pim_matrix_handle_t *handle, int16_t rows, int16_t cols, simplepim_management_t* management);

/**
 * @brief Free a PIM matrix handle and associated resources.
 * @param handle Pointer to the PIM matrix handle to free.
 */
void free_pim_matrix_handle(pim_matrix_handle_t* handle, simplepim_management_t* management);

/**
 * @brief Add two matrices in PIM.
 * @param handle1 Pointer to the first PIM matrix handle.
 * @param handle2 Pointer to the second PIM matrix handle.
 * @return Pointer to a new PIM matrix handle representing the sum.
 */
pim_matrix_handle_t* add_pim_matrices(const pim_matrix_handle_t* handle1, const pim_matrix_handle_t* handle2, simplepim_management_t* management);

/**
 * @brief Multiply two matrices in PIM.
 * @param handle1 Pointer to the first PIM matrix handle.
 * @param handle2 Pointer to the second PIM matrix handle.
 * @return Pointer to a new PIM matrix handle representing the product.
 */
pim_matrix_handle_t* multiply_pim_matrices(const pim_matrix_handle_t* handle1, const pim_matrix_handle_t* handle2, simplepim_management_t* management);

#endif // __PIM_MATRIX_HANDLE_H___