#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "test_assertions.h"

#include "matrix.h"
#include "pim_matrix_multiplication_frame.h"

Matrix* host_multiply_matrices(const Matrix* matrix1, const Matrix* matrix2) {
    if (!matrix1 || !matrix2 || matrix1->cols != matrix2->rows) return NULL;
    uint16_t * result_data_row_major = malloc(matrix1->rows * matrix2->cols * sizeof(uint16_t));
    if (!result_data_row_major) return NULL;
    for (int i = 0; i < matrix1->rows; i++) {
        for (int j = 0; j < matrix2->cols; j++) {
            uint16_t sum = 0;
            for (int k = 0; k < matrix1->cols; k++) {
                uint8_t val1, val2;
                matrix_get(matrix1, i, k, &val1);
                matrix_get(matrix2, k, j, &val2);
                sum += val1 * val2;
            }
            result_data_row_major[i*matrix2->cols + j] = sum;
        }
    }
    Matrix* result = matrix_create_from_row_major_array(matrix1->rows, matrix2->cols, result_data_row_major, sizeof(uint16_t));
    if (!result) {
        free(result_data_row_major);
        return NULL;
    }
    free(result_data_row_major);
    return result;
}

Matrix*  dpu_multiply_matrices(Matrix* matrix1, Matrix* matrix2) {
    // Create a sample matrix multiplication frame
    pim_matrix_multiplication_frame_t* frame = create_pim_matrix_multiplication_frame(4, 0, matrix1->rows, matrix1->cols, matrix2->rows, matrix2->cols, matrix1->rows, matrix2->cols,
                                                                                      sizeof(int8_t), sizeof(int8_t), sizeof(uint16_t));                                                                                  
    ASSERT_TRUE(frame != NULL, "Frame creation failed");
    ASSERT_EQ(frame->num_work_groups, 2, "Frame should have correct number of work groups");
    ASSERT_EQ(frame->work_group_size, 2, "Frame should have correct work group size");
    pim_matrix_multiplication_frame_load_first_matrix(frame, matrix1);
    pim_matrix_multiplication_frame_load_second_matrix(frame, matrix2);
    pim_matrix_multiplication_frame_execute(frame);
    Matrix* result = pim_matrix_multiplication_frame_get_result(frame);
    ASSERT_TRUE(result != NULL, "Result matrix should not be NULL");
    return result;
}

int test_pim_identity_square_matrix_multiplication() {
    printf("Running test_pim_identity_square_matrix_multiplication...\n");
    // Create two sample matrices 16x16
    uint16_t rows = 16, cols = 16;
    uint8_t data1[16*16], data2[16*16];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data1[i*cols + j] = 1; // Sample data for matrix 1
            if (i == j) {
                data2[i*cols + j] = 1; // Sample data for matrix 2
            } else {
                data2[i*cols + j] = 0; // Sample data for matrix 2
            }
        }
    }
    Matrix* matrix1 = matrix_create_from_row_major_array(rows, cols, (void*)data1, sizeof(uint8_t));
    Matrix* matrix2 = matrix_create_from_row_major_array(rows, cols, (void*)data2, sizeof(uint8_t));
    ASSERT_TRUE(matrix1 != NULL, "Matrix 1 creation failed");
    ASSERT_TRUE(matrix2 != NULL, "Matrix 2 creation failed");
    Matrix* result = dpu_multiply_matrices(matrix1, matrix2);
    Matrix* expected_result = host_multiply_matrices(matrix1, matrix2);
    printf("Expected dimensions: %dx%d, Result dimensions: %dx%d\n",
           expected_result->rows, expected_result->cols, result->rows, result->cols);
    printf("Expected matrix:\n");
    for (int i = 0; i < expected_result->rows; i++) {
        for (int j = 0; j < expected_result->cols; j++) {
            int16_t val;
            matrix_get(expected_result, i, j, &val);
            printf("| %d ", val);
        }
        printf("|\n");
    }
    printf("Result matrix:\n");
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            int16_t val;
            matrix_get(result, i, j, &val);
            printf("| %d ", val);
        }
        printf("|\n");
    }
    ASSERT_TRUE(result != NULL, "Result matrix should not be NULL");
    ASSERT_TRUE(matrix_compare(result, expected_result), "Result matrix should match expected result");
    matrix_free(matrix1);
    matrix_free(matrix2);
    return 0;
}

int test_pim_square_matrix_multiplication() {
    printf("Running test_pim_square_matrix_multiplication...\n");
    // Create two sample matrices 16x16
    uint16_t rows = 16, cols = 16;
    uint8_t data1[16*16], data2[16*16];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data1[i*cols + j] = i+j;
            data2[i*cols + j] = i+j; 
        }
    }
    Matrix* matrix1 = matrix_create_from_row_major_array(rows, cols, (void*)data1, sizeof(uint8_t));
    Matrix* matrix2 = matrix_create_from_row_major_array(rows, cols, (void*)data2, sizeof(uint8_t));
    ASSERT_TRUE(matrix1 != NULL, "Matrix 1 creation failed");
    ASSERT_TRUE(matrix2 != NULL, "Matrix 2 creation failed");
    Matrix* result = dpu_multiply_matrices(matrix1, matrix2);
    Matrix* expected_result = host_multiply_matrices(matrix1, matrix2);
    printf("Expected dimensions: %dx%d, Result dimensions: %dx%d\n",
           expected_result->rows, expected_result->cols, result->rows, result->cols);
    printf("Expected matrix:\n");
    for (int i = 0; i < expected_result->rows; i++) {
        for (int j = 0; j < expected_result->cols; j++) {
            int16_t val;
            matrix_get(expected_result, i, j, &val);
            printf("| %d ", val);
        }
        printf("|\n");
    }
    printf("Result matrix:\n");
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            int16_t val;
            matrix_get(result, i, j, &val);
            printf("| %d ", val);
        }
        printf("|\n");
    }
    ASSERT_TRUE(result != NULL, "Result matrix should not be NULL");
    ASSERT_TRUE(matrix_compare(result, expected_result), "Result matrix should match expected result");
    matrix_free(matrix1);
    matrix_free(matrix2);
    return 0;
}

int main() {
    uint32_t fails = 0;
    printf("Running PIM Matrix Multiplication Frame Unittests...\n");
    fails += test_pim_identity_square_matrix_multiplication();
    fails += test_pim_square_matrix_multiplication();
    if (fails == 0) {
        printf("[PASS] All PIM matrix tests passed!\n");
        return 0;
    } else {
        printf("[FAIL] %d PIM matrix tests failed.\n", fails);
        return 1;
    }
}