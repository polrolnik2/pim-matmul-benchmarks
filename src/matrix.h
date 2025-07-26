/**
 * @file matrix.h
 * @brief Abstract representation of a 2D matrix and basic operations (C version).
 *
 * Provides a struct and functions for creating, accessing, and manipulating 2D matrices of int8_t values.
 * All memory management is explicit; the user is responsible for freeing matrices and arrays returned by functions.
 */
#ifndef MATRIX_H
#define MATRIX_H

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Matrix struct representing a 2D matrix of int8_t values.
 *
 * Data is stored in row-major order.
 */
typedef struct {
    int16_t rows; ///< Number of rows
    int16_t cols; ///< Number of columns
    int8_t** data; ///< Pointer to matrix data (array of pointers to rows)
} Matrix;

/**
 * @brief Create a new matrix from a 2D array.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param data 2D array of int8_t values (array of pointers to rows).
 * @return Pointer to new Matrix, or NULL on failure.
 */
Matrix* matrix_create_from_2d_array(int16_t rows, int16_t cols, int8_t **data);

/**
 * @brief Create a new matrix with specified dimensions from a row majow array.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @return Pointer to new Matrix, or NULL on failure.
 */
Matrix* matrix_create_from_row_major_array(int16_t rows, int16_t cols, int8_t *data);

Matrix* matrix_create_from_column_major_array(int16_t rows, int16_t cols, int8_t *data);

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
int8_t* matrix_get_row(const Matrix* mat, int r);

/**
 * @brief Get a dynamically allocated array containing a specific column.
 * @param mat Pointer to Matrix.
 * @param c Column index.
 * @return Pointer to new column array (caller must free), or NULL if out of bounds.
 */
int8_t* matrix_get_col(const Matrix* mat, int c);

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
 * @return Pointer to new string (caller must free), or NULL on failure.
 */
char* matrix_sprint(const Matrix* mat);

/**
 * @brief Print the matrix to stdout.
 * @param mat Pointer to Matrix.
 */
void matrix_print(const Matrix* mat);

/**
 * @brief Access an element in the matrix.
 * @param mat Pointer to Matrix.
 * @param r Row index.
 * @param c Column index.
 * @return Value at (r, c), or 0 if out of bounds.
 */
int8_t matrix_get(const Matrix* mat, int r, int c);

/**
 * @brief Set an element in the matrix.
 * @param mat Pointer to Matrix.
 * @param r Row index.
 * @param c Column index.
 * @param value Value to set.
 * @return 0 on success, -1 if out of bounds.
 */
int matrix_set(Matrix* mat, int r, int c, int8_t value);

#endif // MATRIX_H