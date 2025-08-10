#include <string.h>

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "processing/zip/Zip.h"
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
    uint8_t* matrix_data = malloc_broadcast_aligned(1, matrix->rows * matrix->cols * sizeof(int8_t), management);
    for (int r = 0; r < matrix->rows; ++r) {
        memcpy(matrix_data + r * matrix->cols, matrix->data[r], matrix->cols * sizeof(int8_t));
    }
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

/**
 * @brief Enhanced matrix multiplication using custom DPU memory manager
 * 
 * This function performs matrix multiplication using the custom DPU kernel with
 * the thread memory manager for optimal tasklet distribution and memory management.
 *
 * @param handle1 First matrix handle
 * @param handle2 Second matrix handle  
 * @param management SimplePIM management context
 * @return New handle for result matrix, or NULL on failure
 */
pim_matrix_handle_t* multiply_pim_matrices(const pim_matrix_handle_t* handle1, const pim_matrix_handle_t* handle2,
                                                    simplepim_management_t* management) {
    if (!handle1 || !handle2 || !management) return NULL;

    // Validate dimensions for matrix multiplication
    if (handle1->submatrix_cols != handle2->submatrix_rows) {
        printf("Error: Matrix dimensions incompatible for multiplication (%u x %u) * (%u x %u)\n",
               handle1->submatrix_rows, handle1->submatrix_cols, handle2->submatrix_rows, handle2->submatrix_cols);
        return NULL;
    }
    
    // Create result handle  
    pim_matrix_handle_t* result_handle = (pim_matrix_handle_t*)malloc(sizeof(pim_matrix_handle_t));
    if (!result_handle) return NULL;
    
    result_handle->pim_handle = strdup(pim_id_generate_unique_handle("enhanced_multiply_result"));
    result_handle->submatrix_rows = handle1->submatrix_rows;
    result_handle->submatrix_cols = handle2->submatrix_cols;

    // Prepare DPU arguments structure
    typedef struct {
        uint32_t matrix1_start_offset;
        uint32_t matrix2_start_offset;
        uint32_t result_start_offset;
        uint32_t matrix1_rows;
        uint32_t matrix1_cols;
        uint32_t matrix2_rows;
        uint32_t matrix2_cols;
        uint32_t result_rows;
        uint32_t result_cols;
        uint32_t matrix1_type_size;
        uint32_t matrix2_type_size;
        uint32_t result_type_size;
    } matrix_multiply_arguments_t;
    
    // Get DPU set and number of DPUs from management
    struct dpu_set_t set = management->set;
    uint32_t num_dpus = management->num_dpus;
    
    // Load the custom matrix multiplication DPU binary
    const char* dpu_binary = "/workspace/bin/matrix_multiply_dpu";
    DPU_ASSERT(dpu_load(set, dpu_binary, NULL));
    
    // Prepare arguments for each DPU
    matrix_multiply_arguments_t* input_args = malloc(num_dpus * sizeof(matrix_multiply_arguments_t));
    if (!input_args) {
        free(result_handle->pim_handle);
        free(result_handle);
        return NULL;
    }
    
    // Get table offsets for the matrices
    table_host_t* table1 = lookup_table(handle1->pim_handle, management);
    table_host_t* table2 = lookup_table(handle2->pim_handle, management);
    
    if (!table1 || !table2) {
        printf("ERROR: Failed to lookup tables - table1=%p, table2=%p\n", table1, table2);
        free(input_args);
        free(result_handle->pim_handle);
        free(result_handle);
        return NULL;
    }
    
    // Debug: Print table information
    printf("DEBUG: Table1 - name=%s, start=%u, end=%u, len=%u, type_size=%u\n",
           table1->name, table1->start, table1->end, table1->len, table1->table_type_size);
    printf("DEBUG: Table2 - name=%s, start=%u, end=%u, len=%u, type_size=%u\n",
           table2->name, table2->start, table2->end, table2->len, table2->table_type_size);
    
    // Calculate result matrix size and allocate space
    uint32_t result_elements = handle1->submatrix_rows * handle2->submatrix_cols;
    uint32_t result_size_bytes = result_elements * sizeof(uint16_t);
    uint32_t result_offset = management->free_space_start_pos;
    
    // Setup arguments for each DPU
    struct dpu_set_t dpu;
    int i;
    DPU_FOREACH(set, dpu, i) {
        input_args[i].matrix1_start_offset = table1->start;
        input_args[i].matrix2_start_offset = table2->start;
        input_args[i].result_start_offset = result_offset;
        input_args[i].matrix1_rows = handle1->submatrix_rows;
        input_args[i].matrix1_cols = handle1->submatrix_cols;
        input_args[i].matrix2_rows = handle2->submatrix_rows;
        input_args[i].matrix2_cols = handle2->submatrix_cols;
        input_args[i].result_rows = handle1->submatrix_rows;
        input_args[i].result_cols = handle2->submatrix_cols;
        input_args[i].matrix1_type_size = sizeof(int8_t);
        input_args[i].matrix2_type_size = sizeof(int8_t);
        input_args[i].result_type_size = sizeof(uint16_t);
        
        // Debug: Print DPU arguments
        if (i < 2) { // Only print first 2 DPUs to avoid spam
            printf("DEBUG: DPU %d - M1_offset=%u, M2_offset=%u, Result_offset=%u\n",
                   i, input_args[i].matrix1_start_offset, input_args[i].matrix2_start_offset, input_args[i].result_start_offset);
            printf("DEBUG: DPU %d - M1: %ux%u, M2: %ux%u, Result: %ux%u\n",
                   i, input_args[i].matrix1_rows, input_args[i].matrix1_cols,
                   input_args[i].matrix2_rows, input_args[i].matrix2_cols,
                   input_args[i].result_rows, input_args[i].result_cols);
        }
        
        DPU_ASSERT(dpu_prepare_xfer(dpu, input_args + i));
    }
    
    // Transfer arguments to DPUs
    DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "MATRIX_MULTIPLY_ARGUMENTS", 0, 
                            sizeof(matrix_multiply_arguments_t), DPU_XFER_DEFAULT));
    
    // Launch DPU execution
    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));
    
    // Register result table in management
    table_host_t* result_table = malloc(sizeof(table_host_t));
    result_table->name = strdup(result_handle->pim_handle);
    result_table->start = result_offset;
    result_table->end = result_offset + result_size_bytes;
    result_table->len = result_elements;
    result_table->table_type_size = sizeof(uint16_t);
    result_table->is_virtual_zipped = 0;
    
    // Calculate lens for each DPU (uniform distribution)
    uint32_t* lens = malloc(num_dpus * sizeof(uint32_t));
    uint32_t elements_per_dpu = (result_elements + num_dpus - 1) / num_dpus;
    for (int j = 0; j < num_dpus; j++) {
        if ((j + 1) * elements_per_dpu <= result_elements) {
            lens[j] = elements_per_dpu;
        } else if (j * elements_per_dpu < result_elements) {
            lens[j] = result_elements - j * elements_per_dpu;
        } else {
            lens[j] = 0;
        }
    }
    result_table->lens_each_dpu = lens;

    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_log_read(dpu, stdout));
    }
    
    add_table(result_table, management);
    management->free_space_start_pos = result_table->end + (8 - result_table->end % 8);
    
    // Clean up
    free(input_args);
    
    return result_handle;
}

pim_matrix_handle_t* add_pim_matrices(const pim_matrix_handle_t* handle1, const pim_matrix_handle_t* handle2, simplepim_management_t* management) {
    // if (!handle1 || !handle2 || !management) return NULL;
    
    // // Create result handle
    // pim_matrix_handle_t* result_handle = (pim_matrix_handle_t*)malloc(sizeof(pim_matrix_handle_t));
    // if (!result_handle) return NULL;
    
    // result_handle->pim_handle = strdup(pim_id_generate_unique_handle("added_matrix"));
    // result_handle->submatrix_rows = handle1->submatrix_rows;
    // result_handle->submatrix_cols = handle1->submatrix_cols;
    
    // // Use SimplePIM map interface to apply matrix addition kernel
    // handle_t* add_kernel = create_handle("matrix_add_dpu", MAP);
    // if (!add_kernel) {
    //     free(result_handle->pim_handle);
    //     free(result_handle);
    //     return NULL;
    // }
    
    // // Create temporary table for the operation
    // char temp_table_name[64];
    // snprintf(temp_table_name, sizeof(temp_table_name), "temp_add_%s_%s", 
    //          handle1->pim_handle, handle2->pim_handle);
    
    // // Use zip operation to combine the two matrices for the addition kernel
    // table_zip(handle1->pim_handle, handle2->pim_handle, temp_table_name, add_kernel, management);
    
    // // Apply the addition map operation
    // table_map(temp_table_name, result_handle->pim_handle, sizeof(int8_t), add_kernel, management, 0);
    
    // // Clean up temporary table
    // free_table(temp_table_name, management);
    
    // return result_handle;
}