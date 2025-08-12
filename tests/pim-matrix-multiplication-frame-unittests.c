#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "test_assertions.h"

#include "matrix.h"
#include "pim_matrix_multiplication_frame.h"

int test_pim_square_matrix_multiplication_frame_execute() {
    printf("Running test_pim_square_matrix_multiplication_frame_execute...\n");
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
    // Create a sample matrix multiplication frame
    pim_matrix_multiplication_frame_t* frame = create_pim_matrix_multiplication_frame(4, 0, rows, cols, rows, cols, rows, cols,
                                                                                      sizeof(int8_t), sizeof(int8_t), sizeof(uint16_t));                                                                                  
    ASSERT_TRUE(frame != NULL, "Frame creation failed");
    ASSERT_EQ(frame->num_work_groups, 2, "Frame should have correct number of work groups");
    ASSERT_EQ(frame->work_group_size, 2, "Frame should have correct work group size");
    pim_matrix_multiplication_frame_load_first_matrix(frame, matrix1);
    pim_matrix_multiplication_frame_load_second_matrix(frame, matrix2);
    pim_matrix_multiplication_frame_execute(frame);
    Matrix* result = pim_matrix_multiplication_frame_get_result(frame);
    ASSERT_TRUE(result != NULL, "Result matrix should not be NULL");
    ASSERT_EQ(result->rows, rows, "Result matrix should have correct number of rows");
    ASSERT_EQ(result->cols, cols, "Result matrix should have correct number of cols");
    // Check result values
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int16_t expected_value = 0;
            for (int k = 0; k < cols; k++) {    // Matrix multiplication logic
                expected_value += (i + k) * (k + j);
            }
            int16_t result_value;
            matrix_get(result, i, j, &result_value);
            ASSERT_EQ(result_value, 1, "Result matrix has incorrect value");
        }
    }
    matrix_free(matrix1);
    matrix_free(matrix2);
    return 0;
}

int main() {
    uint32_t fails = 0;
    printf("Running PIM Matrix Multiplication Frame Unittests...\n");
    fails += test_pim_square_matrix_multiplication_frame_execute();
    if (fails == 0) {
        printf("[PASS] All PIM matrix tests passed!\n");
        return 0;
    } else {
        printf("[FAIL] %d PIM matrix tests failed.\n", fails);
        return 1;
    }
}