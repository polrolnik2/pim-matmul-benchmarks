/**
 * @file matrix.h
 * @brief Abstract representation of a 2D matrix and basic operations (C version).
 *
 * Provides a struct and functions for creating, accessing, and manipulating 2D matrices of any data type.
 * All memory management is explicit; the user is responsible for freeing matrices and arrays returned by functions.
 */
#ifndef MATRIX_H
#define MATRIX_H

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Matrix struct representing a 2D matrix of any data type.
 *
 * Data is stored in row-major order.
 */
typedef struct {
    int16_t rows;          ///< Number of rows
    int16_t cols;          ///< Number of columns
    void** data;           ///< Pointer to matrix data (array of pointers to rows)
    uint32_t element_size; ///< Size of each element in bytes
} Matrix;

/**
 * @brief Create a new matrix from a 2D array.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param data 2D array of values (array of pointers to rows).
 * @param element_size Size of each element in bytes.
 * @return Pointer to new Matrix, or NULL on failure.
 */
Matrix* matrix_create_from_2d_array(int16_t rows, int16_t cols, void **data, uint32_t element_size);

/**
 * @brief Create a new matrix with specified dimensions from a row major array.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param data Row-major array of values.
 * @param element_size Size of each element in bytes.
 * @return Pointer to new Matrix, or NULL on failure.
 */
Matrix* matrix_create_from_row_major_array(int16_t rows, int16_t cols, void *data, uint32_t element_size);

/**
 * @brief Create a new matrix with specified dimensions from a column major array.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param data Column-major array of values.
 * @param element_size Size of each element in bytes.
 * @return Pointer to new Matrix, or NULL on failure.
 */
Matrix* matrix_create_from_column_major_array(int16_t rows, int16_t cols, void *data, uint32_t element_size);

/**
 * @brief Free the memory used by a Matrix.
 * @param mat Pointer to Matrix to free.
 */
void matrix_free(Matrix* mat);

/**
 * @brief Get a pointer to the start of a specific row.
 * @param mat Pointer to Matrix.
 * @param r Row index.
 * @return Pointer to the row (within mat->data), or NULL if out of bounds.
 */
void* matrix_get_row(const Matrix* mat, int r);

/**
 * @brief Get a dynamically allocated array containing a specific column.
 * @param mat Pointer to Matrix.
 * @param c Column index.
 * @return Pointer to new column array (caller must free), or NULL if out of bounds.
 */
void* matrix_get_col(const Matrix* mat, int c);

/**
 * @brief Get matrix data in row-major order.
 * @param mat Pointer to Matrix.
 * @return Pointer to new row-major array (caller must free), or NULL on failure.
 */
void* matrix_get_data_row_major(const Matrix* mat);

/**
 * @brief Get matrix data in column-major order.
 * @param mat Pointer to Matrix.
 * @return Pointer to new column-major array (caller must free), or NULL on failure.
 */
void* matrix_get_data_column_major(const Matrix* mat);

/**
 * @brief Create a deep copy of a matrix.
 * @param mat Pointer to Matrix.
 * @return Pointer to new Matrix (caller must free), or NULL on failure.
 */
Matrix* matrix_clone(const Matrix* mat);

/**
 * @brief Compare two matrices for equality.
 * @param a Pointer to first Matrix.
 * @param b Pointer to second Matrix.
 * @return true if matrices are equal, false otherwise.
 */
bool matrix_compare(const Matrix* a, const Matrix* b);

/**
 * @brief Return a string representation of the matrix.
 * @param mat Pointer to Matrix.
 * @param format Printf-style format string for each element.
 * @return Pointer to new string (caller must free), or NULL on failure.
 */
char* matrix_sprint(const Matrix* mat, const char* format);

/**
 * @brief Print the matrix to stdout.
 * @param mat Pointer to Matrix.
 * @param format Printf-style format string for each element.
 */
void matrix_print(const Matrix* mat, const char* format);

/**
 * @brief Access an element in the matrix.
 * @param mat Pointer to Matrix.
 * @param r Row index.
 * @param c Column index.
 * @param out Pointer to store the element value.
 * @return 0 on success, -1 if out of bounds.
 */
int matrix_get(const Matrix* mat, int r, int c, void* out);

/**
 * @brief Set an element in the matrix.
 * @param mat Pointer to Matrix.
 * @param r Row index.
 * @param c Column index.
 * @param value Pointer to value to set.
 * @return 0 on success, -1 if out of bounds.
 */
int matrix_set(Matrix* mat, int r, int c, const void* value);

/**
 * @brief Split a matrix into multiple submatrices by rows.
 * @param mat Pointer to Matrix.
 * @param num_submatrices Number of submatrices to create.
 * @return Array of pointers to new submatrices (caller must free each submatrix and the array), or NULL on failure.
 */
Matrix ** matrix_split_by_rows(const Matrix* mat, int num_submatrices);

/**
 * @brief Join multiple submatrices into a single matrix by rows.
 * @param submatrices Array of pointers to submatrices.
 * @param num_submatrices Number of submatrices.
 * @return Pointer to new Matrix (caller must free), or NULL on failure.
 */
Matrix * matrix_join_by_rows(Matrix **submatrices, int num_submatrices);

/**
 * @brief Split a matrix into multiple submatrices by columns.
 * @param mat Pointer to Matrix.
 * @param num_submatrices Number of submatrices to create.
 * @return Array of pointers to new submatrices (caller must free each submatrix and the array), or NULL on failure.
 */
Matrix ** matrix_split_by_cols(const Matrix* mat, int num_submatrices);

/**
 * @brief Join multiple submatrices into a single matrix by columns.
 * @param submatrices Array of pointers to submatrices.
 * @param num_submatrices Number of submatrices.
 * @return Pointer to new Matrix (caller must free), or NULL on failure.
 */
Matrix * matrix_join_by_cols(Matrix **submatrices, int num_submatrices);

// Type-safe helper macros for common data types
#define MATRIX_CREATE_INT8(rows, cols, data) matrix_create_from_2d_array(rows, cols, (void**)data, sizeof(int8_t))
#define MATRIX_CREATE_INT16(rows, cols, data) matrix_create_from_2d_array(rows, cols, (void**)data, sizeof(int16_t))
#define MATRIX_CREATE_INT32(rows, cols, data) matrix_create_from_2d_array(rows, cols, (void**)data, sizeof(int32_t))
#define MATRIX_CREATE_FLOAT(rows, cols, data) matrix_create_from_2d_array(rows, cols, (void**)data, sizeof(float))
#define MATRIX_CREATE_DOUBLE(rows, cols, data) matrix_create_from_2d_array(rows, cols, (void**)data, sizeof(double))

#define MATRIX_CREATE_FROM_ARRAY_INT8(rows, cols, data) matrix_create_from_row_major_array(rows, cols, data, sizeof(int8_t))
#define MATRIX_CREATE_FROM_ARRAY_INT16(rows, cols, data) matrix_create_from_row_major_array(rows, cols, data, sizeof(int16_t))
#define MATRIX_CREATE_FROM_ARRAY_INT32(rows, cols, data) matrix_create_from_row_major_array(rows, cols, data, sizeof(int32_t))
#define MATRIX_CREATE_FROM_ARRAY_FLOAT(rows, cols, data) matrix_create_from_row_major_array(rows, cols, data, sizeof(float))
#define MATRIX_CREATE_FROM_ARRAY_DOUBLE(rows, cols, data) matrix_create_from_row_major_array(rows, cols, data, sizeof(double))

#define MATRIX_GET_TYPED(mat, r, c, type) \
    ({ type result; matrix_get(mat, r, c, &result) == 0 ? result : (type)0; })

#define MATRIX_SET_TYPED(mat, r, c, value, type) \
    ({ type temp = (value); matrix_set(mat, r, c, &temp); })

#endif // MATRIX_H