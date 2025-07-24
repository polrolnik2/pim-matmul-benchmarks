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

pim_matrix_handle_t* broadcast_matrix_to_pim(const Matrix* matrix, simplepim_management_t* management) {
    if (!matrix || !management) return NULL;
    pim_matrix_handle_t* handle = (pim_matrix_handle_t*)malloc(sizeof(pim_matrix_handle_t));
    if (!handle) return NULL;
    int8_t* matrix_data = malloc_broadcast_aligned(1, matrix->rows * matrix->cols * sizeof(int8_t), management);
    memcpy(matrix_data, matrix->data, matrix->rows * matrix->cols * sizeof(int8_t));
    // Set the PIM handle to a unique identifier (e.g., matrix ID)
    handle->pim_handle = strdup("broadcasted_matrix");
    handle->submatrix_rows = matrix->rows;
    handle->submatrix_cols = matrix->cols;
    // Broadcast the matrix data to all PIM units
    simplepim_broadcast(handle->pim_handle, matrix_data, 1, handle->submatrix_rows * handle->submatrix_cols * sizeof(int8_t), management);
    free(matrix_data);
    return handle;
}

Matrix* gather_matrix_from_pim(pim_matrix_handle_t *handle, int16_t rows, int16_t cols, simplepim_management_t* management) {
    if (!handle || rows <= 0 || cols <= 0 || !management) return NULL;
    // Gather the matrix data from PIM
    int8_t* gathered_data = simplepim_gather(handle->pim_handle, management);
    if (!gathered_data) {
        return NULL;
    }
    Matrix* mat = matrix_create_from_row_major_array(rows, cols, gathered_data);
    free(gathered_data);
    return mat;
}