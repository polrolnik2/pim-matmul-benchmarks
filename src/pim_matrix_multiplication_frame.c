#include <string.h>

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include <dpu.h>

#include <math.h>

#include <matrix.h>

#include "dpu_pim_matrix_multiply_kernel_arguments.h"

#include "pim_matrix_multiplication_frame.h"

uint32_t calculate_pad_rows(int16_t rows, int16_t element_size) {
    uint32_t col_size = rows * element_size;
    uint32_t pad = (8 - (col_size % 8)) % 8;
    return pad / element_size;
}

uint32_t calculate_pad_cols(int16_t cols, int16_t element_size) {
    uint32_t row_size = cols * element_size;
    uint32_t pad = (8 - (row_size % 8)) % 8;
    return pad / element_size;
}

Matrix * matrix_align(const Matrix *mat) {
    if (!mat) return NULL;
    int16_t pad_rows = calculate_pad_rows(mat->rows, mat->element_size);
    int16_t pad_cols = calculate_pad_cols(mat->cols, mat->element_size);
    Matrix *aligned = matrix_add_cols(mat, pad_cols, NULL);
    if (!aligned) {
        return NULL;
    }
    aligned = matrix_add_rows(aligned, pad_rows, NULL);
    if (!aligned) {
        matrix_free(aligned);
        return NULL;
    }
    return aligned;
}

static void find_optimal_work_group_config(uint32_t num_dpus, uint32_t matrix1_size, uint32_t matrix2_size,
                                          uint32_t* num_work_groups, uint32_t* work_group_size) {
    double best_cost = INFINITY;
    uint32_t best_num_work_groups = 1;
    uint32_t best_work_group_size = num_dpus;
    // Try all divisors of num_dpus to find the optimal configuration
    for (uint32_t nwg = 1; nwg * nwg <= num_dpus; nwg++) {
        if (num_dpus % nwg != 0) continue;
        uint32_t wgs = num_dpus / nwg;
        // Check both (nwg, wgs) and (wgs, nwg) if they are different
        double cost = (double)matrix2_size / nwg + (double)matrix1_size / wgs;
        if (cost < best_cost) {
            best_cost = cost;
            best_num_work_groups = nwg;
            best_work_group_size = wgs;
        }
    }
    *num_work_groups = best_num_work_groups;
    *work_group_size = best_work_group_size;
}

pim_matrix_multiplication_frame_t* create_pim_matrix_multiplication_frame(uint32_t num_dpus, uint32_t dpu_offset,
                                                                        uint32_t matrix1_rows, uint32_t matrix1_cols,
                                                                        uint32_t matrix2_rows, uint32_t matrix2_cols,
                                                                        uint32_t result_rows, uint32_t result_cols,
                                                                        uint32_t matrix1_type_size, uint32_t matrix2_type_size, uint32_t result_type_size) {
    pim_matrix_multiplication_frame_t* frame = (pim_matrix_multiplication_frame_t*)malloc(sizeof(pim_matrix_multiplication_frame_t));
    if (!frame) {
        return NULL;
    }

    struct dpu_set_t set;
    DPU_ASSERT(dpu_alloc(num_dpus, NULL, &set));
    frame->dpu_set = set;

    uint32_t matrix1_size = matrix1_rows * matrix1_cols * matrix1_type_size;
    uint32_t matrix2_size = matrix2_rows * matrix2_cols * matrix2_type_size;

    // Find optimal work group configuration with round numbers
    uint32_t optimal_num_work_groups, optimal_work_group_size;
    find_optimal_work_group_config(num_dpus, matrix1_size, matrix2_size, 
                                  &optimal_num_work_groups, &optimal_work_group_size);
    
    frame->num_work_groups = optimal_num_work_groups;
    frame->work_group_size = optimal_work_group_size;
    frame->num_dpus = num_dpus;
    frame->matrix1_rows = matrix1_rows;
    frame->matrix1_cols = matrix1_cols;

    frame->matrix2_rows = matrix2_rows;
    frame->matrix2_cols = matrix2_cols;

    frame->result_rows = matrix1_rows;
    frame->result_cols = matrix2_cols;

    frame->matrix1_type_size = matrix1_type_size;
    frame->matrix2_type_size = matrix2_type_size;
    frame->result_type_size = result_type_size;

    uint32_t curr_offset = dpu_offset;
    frame->matrix1_start_offset = curr_offset;

    uint32_t matrix1_rows_aligned = matrix1_rows + (frame->work_group_size - (matrix1_rows % frame->work_group_size)) % frame->work_group_size;
    uint32_t matrix1_rows_transfer_aligned = matrix1_rows_aligned + calculate_pad_rows(matrix1_rows_aligned, frame->matrix1_type_size);
    uint32_t matrix1_cols_transfer_aligned = matrix1_cols + calculate_pad_cols(matrix1_cols, frame->matrix1_type_size);
    uint32_t matrix1_size_aligned = matrix1_rows_transfer_aligned * matrix1_cols_transfer_aligned * matrix1_type_size;
    curr_offset += matrix1_size_aligned / frame->num_work_groups;
    frame->matrix2_start_offset = curr_offset;

    uint32_t matrix2_cols_aligned = matrix2_cols + (frame->num_work_groups - (matrix2_cols % frame->num_work_groups)) % frame->num_work_groups;
    uint32_t matrix2_rows_transfer_aligned = matrix2_rows + calculate_pad_rows(matrix2_rows, frame->matrix2_type_size);
    uint32_t matrix2_cols_transfer_aligned = matrix2_cols_aligned + calculate_pad_cols(matrix2_cols_aligned, frame->matrix2_type_size);
    uint32_t matrix2_size_aligned = matrix2_rows_transfer_aligned * matrix2_cols_transfer_aligned * matrix2_type_size;
    curr_offset += matrix2_size_aligned / frame->num_work_groups;
    frame->result_start_offset = curr_offset;

    uint32_t result_rows_transfer_aligned = matrix1_rows_aligned + calculate_pad_rows(matrix1_rows_aligned, frame->result_type_size);
    uint32_t result_cols_transfer_aligned = matrix2_cols_aligned + calculate_pad_cols(matrix2_cols_aligned, frame->result_type_size);
    curr_offset += result_rows_transfer_aligned * result_cols_transfer_aligned * frame->result_type_size / frame->num_dpus;
    frame->mem_frame_end = curr_offset;

    frame->result_valid = false;

    const char* dpu_binary = "/workspace/bin/matrix_multiply_dpu";
    DPU_ASSERT(dpu_load(frame->dpu_set, dpu_binary, NULL));

    return frame;
}

void destroy_pim_matrix_multiplication_frame(pim_matrix_multiplication_frame_t* frame);

void pim_matrix_multiplication_frame_load_first_matrix(pim_matrix_multiplication_frame_t* frame, Matrix * matrix) {
    if (!frame || !matrix) return;
    // Load first matrix into MRAM at the specified offset
    uint32_t aligned_rows = (frame->work_group_size - (frame->matrix1_rows % frame->work_group_size)) % frame->work_group_size;
    Matrix * matrix_split_aligned = matrix_add_rows(matrix, aligned_rows, NULL);
    if (!matrix_split_aligned) {
        fprintf(stderr, "Failed to align matrix rows for PIM frame\n");
        return;
    }
    Matrix ** submatrices = matrix_split_by_rows(matrix_split_aligned, frame->work_group_size);
    if (!submatrices) {
        fprintf(stderr, "Failed to split matrix by rows for PIM frame\n");
        matrix_free(matrix_split_aligned);
        return;
    }
    void ** submatrices_data = (void**)malloc(frame->work_group_size * sizeof(void*));
    if (!submatrices_data) {
        fprintf(stderr, "Failed to allocate memory for submatrices data\n");
        free(submatrices);
        matrix_free(matrix_split_aligned);
        return;
    }
    bool * submatrices_data_populated = (bool*)malloc(frame->work_group_size * sizeof(bool));
    if (!submatrices_data_populated) {
        fprintf(stderr, "Failed to allocate memory for submatrices data populated flags\n");
        free(submatrices_data);
        free(submatrices);
        matrix_free(matrix_split_aligned);
        return;
    }
    for (uint32_t i = 0; i < frame->work_group_size; i++) {
        submatrices_data_populated[i] = false;
    }
    // Load the aligned matrix into MRAM
    uint32_t i;
    struct dpu_set_t dpu;
    DPU_FOREACH(frame->dpu_set, dpu, i) {
        if (!submatrices_data_populated[i % frame->work_group_size]) {
            submatrices[i % frame->work_group_size] = matrix_align(submatrices[i % frame->work_group_size]);
            printf("Aligned submatrix %d for PIM frame\n%s", i % frame->work_group_size, matrix_sprint(submatrices[i % frame->work_group_size], "| %u |"));
            if (!submatrices[i % frame->work_group_size]) {
                fprintf(stderr, "Failed to align submatrix for PIM frame\n");
                free(submatrices_data_populated);
                free(submatrices_data);
                free(submatrices);
                matrix_free(matrix_split_aligned);
                return;
            }
            submatrices_data[i % frame->work_group_size] = matrix_get_data_row_major(submatrices[i % frame->work_group_size]);
            if (!submatrices_data[i % frame->work_group_size]) {
                fprintf(stderr, "Failed to get row major data from submatrix\n");
                free(submatrices_data_populated);
                free(submatrices_data);
                free(submatrices);
                matrix_free(matrix_split_aligned);
                return;
            }
            submatrices_data_populated[i % frame->work_group_size] = true;
        }
        DPU_ASSERT(dpu_prepare_xfer(dpu, submatrices_data[i % frame->work_group_size]));
    }
    uint32_t offset = frame->matrix1_start_offset;
    uint32_t submatrix_size = submatrices[0]->rows * submatrices[0]->cols * frame->matrix1_type_size;
    DPU_ASSERT(dpu_push_xfer(frame->dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, offset, submatrix_size, DPU_XFER_DEFAULT));
    free(submatrices_data_populated);
    free(submatrices_data);
    free(submatrices);
    matrix_free(matrix_split_aligned);
    frame->result_valid = false; // Reset result validity after loading new matrix
}

void pim_matrix_multiplication_frame_load_second_matrix(pim_matrix_multiplication_frame_t* frame, Matrix * matrix) {
    if (!frame || !matrix) return;
    // Load second matrix into MRAM at the specified offset
    uint32_t aligned_cols = (frame->num_work_groups - (frame->matrix2_cols % frame->num_work_groups)) % frame->num_work_groups;
    Matrix * matrix_split_aligned = matrix_add_cols(matrix, aligned_cols, NULL);
    if (!matrix_split_aligned) {
        fprintf(stderr, "Failed to align matrix cols for PIM frame\n");
        return;
    }
    Matrix ** submatrices = matrix_split_by_cols(matrix_split_aligned, frame->num_work_groups);
    if (!submatrices) {
        fprintf(stderr, "Failed to split matrix by cols for PIM frame\n");
        matrix_free(matrix_split_aligned);
        return;
    }
    void ** submatrices_data = (void**)malloc(frame->num_work_groups * sizeof(void*));
    if (!submatrices_data) {
        fprintf(stderr, "Failed to allocate memory for submatrices data\n");
        free(submatrices);
        matrix_free(matrix_split_aligned);
        return;
    }
    bool * submatrices_data_populated = (bool*)malloc(frame->num_work_groups * sizeof(bool));
    if (!submatrices_data_populated) {
        fprintf(stderr, "Failed to allocate memory for submatrices data populated flags\n");
        free(submatrices_data);
        free(submatrices);
        matrix_free(matrix_split_aligned);
        return;
    }
    for (uint32_t i = 0; i < frame->num_work_groups; i++) {
        submatrices_data_populated[i] = false;
    }
    // Load the aligned matrix into MRAM
    uint32_t i;
    struct dpu_set_t dpu;
    DPU_FOREACH(frame->dpu_set, dpu, i) {
        if (!submatrices_data_populated[i / frame->work_group_size]) {
            submatrices[i / frame->work_group_size] = matrix_align(submatrices[i / frame->work_group_size]);
            if (!submatrices[i / frame->work_group_size]) {
                fprintf(stderr, "Failed to align submatrix for PIM frame\n");
                free(submatrices_data_populated);
                free(submatrices_data);
                free(submatrices);
                matrix_free(matrix_split_aligned);
                return;
            }
            submatrices_data[i / frame->work_group_size] = matrix_get_data_column_major(submatrices[i / frame->work_group_size]);
            if (!submatrices_data[i / frame->work_group_size]) {
                fprintf(stderr, "Failed to get column major data from submatrix\n");
                free(submatrices_data_populated);
                free(submatrices_data);
                free(submatrices);
                matrix_free(matrix_split_aligned);
                return;
            }
            submatrices_data_populated[i / frame->work_group_size] = true;
        }
        DPU_ASSERT(dpu_prepare_xfer(dpu, submatrices_data[i / frame->work_group_size]));
    }
    uint32_t offset = frame->matrix2_start_offset;
    uint32_t submatrix_size = submatrices[0]->rows * submatrices[0]->cols * frame->matrix2_type_size;
    DPU_ASSERT(dpu_push_xfer(frame->dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, offset, submatrix_size, DPU_XFER_DEFAULT));
    free(submatrices_data_populated);
    free(submatrices_data);
    free(submatrices);
    matrix_free(matrix_split_aligned);
    frame->result_valid = false; // Reset result validity after loading new matrix
}

void pim_matrix_multiplication_frame_execute(pim_matrix_multiplication_frame_t* frame) {
    dpu_pim_matrix_multiply_kernel_arguments_t input_args;
    struct dpu_set_t dpu;
    input_args.matrix1_start_offset = frame->matrix1_start_offset;
    input_args.matrix2_start_offset = frame->matrix2_start_offset;
    input_args.result_start_offset = frame->result_start_offset;
    uint32_t matrix1_split_rows = (frame->result_rows + ((frame->work_group_size - (frame->result_rows % frame->work_group_size)) % frame->work_group_size)) / frame->work_group_size;
    input_args.matrix1_rows = calculate_pad_rows(matrix1_split_rows, frame->matrix1_type_size) + matrix1_split_rows;
    input_args.matrix1_cols = calculate_pad_cols(frame->matrix1_cols, frame->matrix1_type_size) + frame->matrix1_cols;
    input_args.matrix2_cols = calculate_pad_rows(frame->matrix2_rows, frame->matrix2_type_size) + frame->matrix2_rows;
    uint32_t matrix2_split_cols = (frame->matrix2_cols + ((frame->num_work_groups - (frame->matrix2_cols % frame->num_work_groups)) % frame->num_work_groups)) / frame->num_work_groups;
    input_args.matrix2_rows = calculate_pad_cols(matrix2_split_cols, frame->matrix2_type_size) + matrix2_split_cols;
    uint32_t result_rows_frame_aligned = ((frame->result_rows + (frame->work_group_size - (frame->result_rows % frame->work_group_size)) % frame->work_group_size)) / frame->work_group_size;
    uint32_t result_cols_frame_aligned = ((frame->result_cols + (frame->num_work_groups - (frame->result_cols % frame->num_work_groups)) % frame->num_work_groups)) / frame->num_work_groups;
    input_args.result_rows = result_rows_frame_aligned + calculate_pad_rows(result_rows_frame_aligned, frame->result_type_size);
    input_args.result_cols = result_cols_frame_aligned + calculate_pad_cols(result_cols_frame_aligned, frame->result_type_size);
    input_args.matrix1_type_size = frame->matrix1_type_size;
    input_args.matrix2_type_size = frame->matrix2_type_size;
    input_args.result_type_size = frame->result_type_size;

    DPU_FOREACH(frame->dpu_set, dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &input_args));
    }

    DPU_ASSERT(dpu_push_xfer(frame->dpu_set, DPU_XFER_TO_DPU, "MATRIX_MULTIPLY_ARGUMENTS", 0,
                            sizeof(dpu_pim_matrix_multiply_kernel_arguments_t), DPU_XFER_DEFAULT));

    DPU_ASSERT(dpu_launch(frame->dpu_set, DPU_SYNCHRONOUS));

    // #ifdef DEBUG
    DPU_FOREACH(frame->dpu_set, dpu) {
        DPU_ASSERT(dpu_log_read(dpu, stdout));
    }
    // #endif // DEBUG

    frame->result_valid = true; // Mark result as valid after execution
    return;
}

Matrix * pim_matrix_multiplication_frame_get_result(pim_matrix_multiplication_frame_t* frame) {
    if (!frame) {
        fprintf(stderr, "Frame is NULL\n");
        return NULL;
    }
    
    void *** submatrices_data = (void**)malloc(frame->work_group_size * sizeof(void*));
    if (!submatrices_data) {
        fprintf(stderr, "Failed to allocate memory for submatrices data\n");
        return NULL;
    }
    bool * submatrices_row_populated = (bool*)malloc(frame->work_group_size * sizeof(bool));
    if (!submatrices_row_populated) {
        fprintf(stderr, "Failed to allocate memory for submatrices row populated flags\n");
        free(submatrices_data);
        return NULL;
    }

    uint32_t result_rows_frame_aligned = ((frame->result_rows + (frame->work_group_size - (frame->result_rows % frame->work_group_size)) % frame->work_group_size)) / frame->work_group_size;
    uint32_t result_cols_frame_aligned = ((frame->result_cols + (frame->num_work_groups - (frame->result_cols % frame->num_work_groups)) % frame->num_work_groups)) / frame->num_work_groups;
    uint32_t result_rows_dpu_transfer_aligned = result_rows_frame_aligned + calculate_pad_rows(result_rows_frame_aligned, frame->result_type_size);
    uint32_t result_cols_dpu_transfer_aligned = result_cols_frame_aligned + calculate_pad_cols(result_cols_frame_aligned, frame->result_type_size);
    uint32_t result_size_aligned = result_rows_dpu_transfer_aligned * result_cols_dpu_transfer_aligned * frame->matrix2_type_size;
    printf("Result matrix size: %u rows, %u cols, %u type size, total size: %u bytes\n",
           result_rows_frame_aligned, result_cols_frame_aligned, frame->result_type_size, result_size_aligned);
    for (uint32_t i = 0; i < frame->work_group_size; i++) {
        submatrices_row_populated[i] = false;
    }
    uint32_t i;
    struct dpu_set_t dpu;
    DPU_FOREACH(frame->dpu_set, dpu, i) {
        uint32_t col = i % frame->work_group_size;
        uint32_t row = i / frame->work_group_size;
        if (!submatrices_row_populated[i % frame->work_group_size]) {
            submatrices_data[i % frame->work_group_size] = malloc(frame->num_work_groups * sizeof(void*));
            if (!submatrices_data[i % frame->work_group_size]) {
                fprintf(stderr, "Failed to allocate memory for submatrix data row\n");
                // Clean up previously allocated memory
                for (uint32_t cleanup_i = 0; cleanup_i < i; cleanup_i++) {
                    if (submatrices_row_populated[cleanup_i % frame->work_group_size]) {
                        free(submatrices_data[cleanup_i % frame->work_group_size]);
                    }
                }
                free(submatrices_row_populated);
                free(submatrices_data);
                return NULL;
            }
            submatrices_row_populated[i % frame->work_group_size] = true;
        }
        submatrices_data[row][col] = malloc(result_rows_dpu_transfer_aligned * result_cols_dpu_transfer_aligned * frame->result_type_size);
        if (!submatrices_data[row][col]) {
            fprintf(stderr, "Failed to allocate memory for submatrix data element\n");
            // Clean up previously allocated memory
            for (uint32_t cleanup_i = 0; cleanup_i <= i; cleanup_i++) {
                if (submatrices_row_populated[cleanup_i % frame->work_group_size]) {
                    free(submatrices_data[cleanup_i % frame->work_group_size]);
                }
            }
            free(submatrices_row_populated);
            free(submatrices_data);
            return NULL;
        }
        DPU_ASSERT(dpu_prepare_xfer(dpu, submatrices_data[row][col]));
    }
    DPU_ASSERT(dpu_push_xfer(frame->dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, frame->result_start_offset,
                            result_rows_dpu_transfer_aligned * result_cols_dpu_transfer_aligned * frame->result_type_size, DPU_XFER_DEFAULT));
    Matrix *** submatrices = (Matrix***)malloc(frame->work_group_size * sizeof(Matrix**));
    if (!submatrices) {
        fprintf(stderr, "Failed to allocate memory for submatrices\n");
        // Clean up submatrices_data
        for (uint32_t cleanup_i = 0; cleanup_i < frame->work_group_size; cleanup_i++) {
            if (submatrices_row_populated[cleanup_i]) {
                free(submatrices_data[cleanup_i]);
            }
        }
        free(submatrices_row_populated);
        free(submatrices_data);
        return NULL;
    }
    Matrix ** row_submatrices = (Matrix**)malloc(frame->work_group_size * sizeof(Matrix*));
    if (!row_submatrices) {
        fprintf(stderr, "Failed to allocate memory for row submatrices\n");
        // Clean up submatrices_data
        for (uint32_t cleanup_i = 0; cleanup_i < frame->work_group_size; cleanup_i++) {
            if (submatrices_row_populated[cleanup_i]) {
                free(submatrices_data[cleanup_i]);
            }
        }
        free(submatrices);
        free(submatrices_row_populated);
        free(submatrices_data);
        return NULL;
    }
    for (uint32_t i = 0; i < frame->work_group_size; i++) {
        submatrices[i] = (Matrix**)malloc(frame->num_work_groups * sizeof(Matrix*));
        if (!submatrices[i]) {
            fprintf(stderr, "Failed to allocate memory for submatrix row %u\n", i);
            // Clean up previously allocated submatrices
            for (uint32_t cleanup_i = 0; cleanup_i < i; cleanup_i++) {
                for (uint32_t cleanup_j = 0; cleanup_j < frame->num_work_groups; cleanup_j++) {
                    if (submatrices[cleanup_i][cleanup_j]) {
                        matrix_free(submatrices[cleanup_i][cleanup_j]);
                    }
                }
                free(submatrices[cleanup_i]);
            }
            // Clean up submatrices_data
            for (uint32_t cleanup_i = 0; cleanup_i < frame->work_group_size; cleanup_i++) {
                if (submatrices_row_populated[cleanup_i]) {
                    free(submatrices_data[cleanup_i]);
                }
            }
            free(row_submatrices);
            free(submatrices);
            free(submatrices_row_populated);
            free(submatrices_data);
            return NULL;
        }
        for (uint32_t j = 0; j < frame->num_work_groups; j++) {
            submatrices[i][j] = matrix_create_from_row_major_array(result_rows_dpu_transfer_aligned, result_cols_dpu_transfer_aligned, submatrices_data[i][j], frame->result_type_size);
            if (!submatrices[i][j]) {
                fprintf(stderr, "Failed to create submatrix from row major array\n");
                // Clean up previously created submatrices in this row
                for (uint32_t cleanup_j = 0; cleanup_j < j; cleanup_j++) {
                    matrix_free(submatrices[i][cleanup_j]);
                }
                // Clean up previously allocated submatrices
                for (uint32_t cleanup_i = 0; cleanup_i < i; cleanup_i++) {
                    for (uint32_t cleanup_j = 0; cleanup_j < frame->num_work_groups; cleanup_j++) {
                        if (submatrices[cleanup_i][cleanup_j]) {
                            matrix_free(submatrices[cleanup_i][cleanup_j]);
                        }
                    }
                    free(submatrices[cleanup_i]);
                }
                free(submatrices[i]);
                // Clean up submatrices_data
                for (uint32_t cleanup_i = 0; cleanup_i < frame->work_group_size; cleanup_i++) {
                    if (submatrices_row_populated[cleanup_i]) {
                        free(submatrices_data[cleanup_i]);
                    }
                }
                free(row_submatrices);
                free(submatrices);
                free(submatrices_row_populated);
                free(submatrices_data);
                return NULL;
            }
            printf("Submatrix %d:%d for PIM frame\n%s", i, j, matrix_sprint(submatrices[i][j], "| %u |"));
            submatrices[i][j] = matrix_extract_submatrix(submatrices[i][j], result_rows_frame_aligned, result_cols_frame_aligned);
            if (!submatrices[i][j]) {
                fprintf(stderr, "Failed to extract submatrix\n");
                free(submatrices_data);
                free(submatrices);
                free(row_submatrices);
                return NULL;
            }
        }
        row_submatrices[i] = matrix_join_by_rows(submatrices[i], frame->work_group_size);
    }
    Matrix * result = matrix_join_by_cols(row_submatrices, frame->num_work_groups);
    if (!result) {
        fprintf(stderr, "Failed to join submatrices by columns\n");
        for (uint32_t i = 0; i < frame->work_group_size; i++) {
            matrix_free(row_submatrices[i]);
            free(submatrices[i]);
        }
        free(row_submatrices);
        free(submatrices);
        free(submatrices_data);
        free(submatrices_row_populated);
        return NULL;
    }
    result = matrix_extract_submatrix(result, frame->result_rows, frame->result_cols);
    return result;
}