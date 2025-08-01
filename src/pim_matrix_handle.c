#include <string.h>

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "processing/map/Map.h"
#include "processing/zip/Zip.h"
#include "processing/gen_red/GenRed.h"
#include "processing/ProcessingHelperHost.h"
#include "communication/CommOps.h"
#include "management/Management.h"

#include "matrix.h"
#include "pim_matrix_handle.h"

char * pim_id_generate_unique_handle(const char *prefix) {
    static int counter = 0; // Static to maintain state across calls
    char *handle = (char *)malloc(64);
    if (!handle) return NULL;
    snprintf(handle, 64, "%s_%d", prefix, counter);
    counter++;
    return handle;
}

pim_matrix_handle_t* broadcast_matrix_to_pim(const Matrix* matrix, simplepim_management_t* management) {
    if (!matrix || !management) return NULL;
    pim_matrix_handle_t* handle = (pim_matrix_handle_t*)malloc(sizeof(pim_matrix_handle_t));
    if (!handle) return NULL;
    int8_t* matrix_data = malloc_broadcast_aligned(1, matrix->rows * matrix->cols * sizeof(int8_t), management);
    for (int r = 0; r < matrix->rows; ++r) {
        memcpy(matrix_data + r * matrix->cols, matrix->data[r], matrix->cols * sizeof(int8_t));
    }
    static int broadcast_matrix_counter = 0;
    handle->pim_handle = strdup(pim_id_generate_unique_handle("broadcasted_matrix"));
    handle->submatrix_rows = matrix->rows;
    handle->submatrix_cols = matrix->cols;
    // Broadcast the matrix data to all PIM units
    simplepim_broadcast(handle->pim_handle, matrix_data, 1, handle->submatrix_rows * handle->submatrix_cols * sizeof(int8_t), management);
    free(matrix_data);
    return handle;
}

pim_matrix_handle_t* scatter_matrix_to_pim(const Matrix* matrix, int16_t submatrix_rows, int16_t submatrix_cols, simplepim_management_t* management) {
    if (!matrix || !management || submatrix_rows <= 0 || submatrix_cols <= 0) return NULL;
    pim_matrix_handle_t* handle = (pim_matrix_handle_t*)malloc(sizeof(pim_matrix_handle_t));
    if (!handle) return NULL;
    static int scatter_matrix_counter = 0;
    handle->pim_handle = strdup(pim_id_generate_unique_handle("scattered_matrix"));
    handle->submatrix_rows = submatrix_rows;
    handle->submatrix_cols = submatrix_cols;
    int submatrices_by_rows = matrix->rows / submatrix_rows;
    int submatrices_by_cols = matrix->cols / submatrix_cols;
    Matrix** submatrices_row = matrix_split_by_rows(matrix, submatrices_by_rows);
    Matrix *** submatrices = malloc(submatrices_by_rows * sizeof(Matrix**));
    if (!submatrices) {
        free(handle);
        free(submatrices_row);
        return NULL;
    }
    for (int i = 0; i < submatrices_by_rows; ++i) {
        submatrices[i] = matrix_split_by_cols(submatrices_row[i], submatrices_by_cols);
        if (!submatrices[i]) {
            for (int j = 0; j < i; ++j) {
                matrix_free(submatrices_row[j]);
            }
            free(submatrices_row);
            free(submatrices);
            free(handle);
            return NULL;
        }
    }
    free(submatrices_row);
    int8_t* scattered_data = malloc_scatter_aligned(1, matrix->rows * matrix->cols * sizeof(int8_t), management);
    if (!scattered_data) {
        for (int i = 0; i < submatrices_by_rows; ++i) {
            for (int j = 0; j < submatrices_by_cols; ++j) {
                matrix_free(submatrices[i][j]);
            }
            free(submatrices[i]);
        }
        free(submatrices);
        free(handle);
        return NULL;
    }
    for (int i = 0; i < submatrices_by_rows; ++i) {
        for (int j = 0; j < submatrices_by_cols; ++j) {
            memcpy(scattered_data + (i * submatrices_by_cols + j) * submatrix_rows * submatrix_cols * sizeof(int8_t),
                   matrix_get_data_row_major(submatrices[i][j]), submatrix_rows * submatrix_cols * sizeof(int8_t));
        }
    }
    simplepim_scatter(handle->pim_handle, scattered_data, submatrices_by_cols * submatrices_by_rows, submatrix_cols * submatrix_rows * sizeof(int8_t), management);
    return handle;
}

Matrix* gather_matrix_from_pim(pim_matrix_handle_t *handle, int16_t rows, int16_t cols, simplepim_management_t* management) {
    if (!handle || rows <= 0 || cols <= 0 || !management) return NULL;
    int submatrix_rows = handle->submatrix_rows;
    int submatrix_cols = handle->submatrix_cols;
    int submatrices_by_rows = rows / submatrix_rows;
    int submatrices_by_cols = cols / submatrix_cols;
    // Gather the matrix data from PIM
    int8_t* gathered_data = simplepim_gather(handle->pim_handle, management);
    if (!gathered_data) {
        return NULL;
    }
    Matrix ** submatrices = malloc(submatrices_by_rows * sizeof(Matrix*));
    for (int i = 0; i < submatrices_by_rows; ++i) {
        Matrix** row_submatrices = malloc(submatrices_by_cols * sizeof(Matrix*));
        for (int j = 0; j < submatrices_by_cols; ++j) {
            row_submatrices[j] = matrix_create_from_row_major_array(submatrix_rows, submatrix_cols,
                gathered_data + (i * submatrices_by_cols + j) * submatrix_rows * submatrix_cols * sizeof(int8_t));
        }
        submatrices[i] = matrix_join_by_cols(row_submatrices, submatrices_by_cols);
        free(row_submatrices);
    }
    Matrix* mat = matrix_join_by_rows(submatrices, submatrices_by_rows);
    free(submatrices);
    free(gathered_data);
    return mat;
}

void free_pim_matrix_handle(pim_matrix_handle_t* handle, simplepim_management_t* management) {
    if (!handle || !management) return;
    // Free the PIM handle and associated resources
    free_table(handle->pim_handle, management);
}